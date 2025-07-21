"""Strategy module for medical federated learning."""

import flwr as fl
import json
import logging
import numpy as np
import pandas as pd
import time
import sys
import os

from collections import defaultdict
from datetime import datetime
from flwr.common import (
    FitRes,
    EvaluateRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
import torch
from torch.utils.data import DataLoader
from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentTracker:
    """Experiment tracking for federated learning rounds."""

    def __init__(self, experiment_name: str, strategy_name: str):
        self.experiment_name = experiment_name
        self.strategy_name = strategy_name
        self.start_time = datetime.now()

        self.round_data: List[Dict[str, Any]] = []
        self.client_data: defaultdict = defaultdict(list)
        self.convergence_metrics: List[Dict[str, Any]] = []

        self.variance_history: List[float] = []
        self.eta_history: List[float] = []
        self.participation_history: defaultdict = defaultdict(int)

    def log_round_start(
            self,
            server_round: int,
            num_selected_clients: int,
            current_eta: float) -> None:
        logger.info(f"Round {server_round} | Clients: {num_selected_clients} | η: {current_eta:.6f}")

    def log_round_complete(
        self,
        server_round: int,
        metrics: Dict[str, Any],
        client_fit_res: List[Tuple[ClientProxy, FitRes]]
    ) -> None:
        round_metrics = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.round_data.append(round_metrics)

        self.variance_history.append(metrics.get("variance", 0.0))
        self.eta_history.append(metrics.get("adapted_eta", 0.0))

        for client, res in client_fit_res:
            self.client_data[client.cid].append({
                "round": server_round,
                **res.metrics
            })
            self.participation_history[client.cid] += 1

        logger.info(f"Round {server_round} - Variance: {metrics.get('variance', 0.0):.6f}, η: {metrics.get('adapted_eta', 0.0):.6f}")

    def export_research_data(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        if output_dir:
            export_dir = Path(output_dir)
        else:
            run_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            export_dir = Path("research_exports") / f"{self.experiment_name}_{run_timestamp}"

        export_dir.mkdir(parents=True, exist_ok=True)
        exported_files = {}

        if self.round_data:
            round_df = pd.DataFrame(self.round_data)
            round_path = export_dir / "server_round_data.csv"
            round_df.to_csv(round_path, index=False)
            exported_files["server_metrics"] = str(round_path)

        if self.client_data:
            all_client_data = []
            for cid, records in self.client_data.items():
                for record in records:
                    all_client_data.append({"client_id": cid, **record})
            client_df = pd.DataFrame(all_client_data)
            client_path = export_dir / "client_aggregated_metrics.csv"
            client_df.to_csv(client_path, index=False)
            exported_files["client_metrics"] = str(client_path)

        logger.info(f"Data exported to {export_dir}")
        return exported_files


class UnifiedFairnessStrategy(Strategy):
    """Unified federated learning strategy with FAUP, AdaFedAdam, and Adaptive Loss."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Metrics]]], Metrics]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[
            List[Tuple[int, Metrics]]], Metrics]] = None,

        eta: float = 1e-2,
        eta_adapt_rate: float = 1.0,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,

        w_impact: float = 0.5,
        w_debt: float = 0.5,

        num_classes: int = 4,
        lambda_val: float = 15.0,

        experiment_name: str = "UnifiedFairness_Experiment",
        checkpoint_dir: Optional[str] = None
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.eta = eta
        self.initial_eta = eta
        self.eta_adapt_rate = eta_adapt_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None

        self.w_impact = w_impact
        self.w_debt = w_debt
        self.client_debts: Dict[str, int] = defaultdict(int)
        self.client_metrics_history: Dict[str, Dict[str, Any]] = {}

        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.kappa_values = np.ones(num_classes) * lambda_val

        self.current_parameters = initial_parameters
        self.experiment_tracker = ExperimentTracker(experiment_name, "UnifiedFairnessStrategy")

        logger.info(f"Strategy initialized - η: {eta}, w_impact: {w_impact}, classes: {num_classes}")

    def initialize_parameters(
            self,
            client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.current_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        available_clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=self.min_available_clients
        )
        if not available_clients:
            return []

        utilities = np.array([
            self._calculate_client_utility(client.cid) for client in available_clients
        ])

        probabilities = utilities / np.sum(utilities) if np.sum(utilities) > 0 else None

        num_to_select = int(self.fraction_fit * len(available_clients))
        num_to_select = max(self.min_fit_clients, num_to_select)

        selected_indices = np.random.choice(
            len(available_clients), size=num_to_select, replace=False, p=probabilities
        )
        selected_clients = [available_clients[i] for i in selected_indices]

        selected_cids = {c.cid for c in selected_clients}
        for client in available_clients:
            if client.cid in selected_cids:
                self.client_debts[client.cid] = 0
            else:
                self.client_debts[client.cid] += 1

        config = {}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)

        config["kappa_values"] = json.dumps(self.kappa_values.tolist())
        fit_ins = fl.common.FitIns(parameters, config)

        return [(client, fit_ins) for client in selected_clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Configure the evaluate round."""
        if self.fraction_evaluate == 0.0:
            return []

        # Sample clients for evaluation
        sample_size = int(self.fraction_evaluate * client_manager.num_available())
        sample_size = max(self.min_evaluate_clients, sample_size)
        
        available_clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients
        )

        config = {}
        if self.on_evaluate_config_fn:
            config = self.on_evaluate_config_fn(server_round)
        
        config["kappa_values"] = json.dumps(self.kappa_values.tolist())
        evaluate_ins = fl.common.EvaluateIns(parameters, config)

        return [(client, evaluate_ins) for client in available_clients]

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters on the server-side."""
        if self.evaluate_fn is None:
            # No server-side evaluation function
            return None
        
        # Use the evaluate function if provided
        eval_result = self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        if eval_result is None:
            return None
        
        loss, metrics = eval_result
        return loss, metrics

    def _calculate_client_utility(self, cid: str) -> float:
        """Calculate utility for a client based on impact and debt."""
        metrics = self.client_metrics_history.get(cid, {})
        loss_before = metrics.get("loss_before", 1.0)
        loss_after = metrics.get("loss_after", 1.0)
        impact = max(0.0, loss_before - loss_after)

        debt = self.client_debts.get(cid, 0)

        return self.w_impact * impact + self.w_debt * debt

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates using AdaFedAdam."""

        if not results:
            return None, {}

        self.experiment_tracker.log_round_start(server_round, len(results), self.eta)

        # Store metrics for next round's FAUP calculation
        for client, fit_res in results:
            self.client_metrics_history[client.cid] = fit_res.metrics

        # --- AdaFedAdam Aggregation ---
        current_weights = parameters_to_ndarrays(self.current_parameters)

        # Calculate client updates and pseudo-gradient
        client_updates = []
        total_examples = sum(res.num_examples for _, res in results)
        for _, res in results:
            update = [
                (cw - sw) * (res.num_examples / total_examples)
                for cw, sw in zip(parameters_to_ndarrays(res.parameters), current_weights)
            ]
            client_updates.append(update)

        pseudo_gradient = [
            -np.sum([u[i] for u in client_updates], axis=0) for i in range(len(current_weights))
        ]

        # Calculate variance based on cosine similarity
        flat_pseudo_gradient = self._flatten_ndarrays(pseudo_gradient)
        flat_client_deltas = [self._flatten_ndarrays(u) for u in client_updates]
        variance, similarities = self._calculate_update_variance(
            flat_client_deltas, flat_pseudo_gradient)

        # Adapt learning rate
        self.eta = self._adapt_learning_rate(variance)

        # Initialize Adam state on first round
        if self.m_t is None:
            self.m_t = [np.zeros_like(p) for p in pseudo_gradient]
            self.v_t = [np.zeros_like(p) for p in pseudo_gradient]

        # Adam updates - ensure m_t and pseudo_gradient are not None
        if self.m_t is not None and pseudo_gradient is not None:
            self.m_t = [self.beta_1 * m + (1 - self.beta_1) * g for m,
                        g in zip(self.m_t, pseudo_gradient)]
        else:
            logger.warning("m_t or pseudo_gradient is None, cannot perform Adam update")
            return None, {}

        # Update v_t with type safety
        if self.v_t is not None and pseudo_gradient is not None:
            self.v_t = [self.beta_2 * v + (1 - self.beta_2) * np.square(g)
                        for v, g in zip(self.v_t, pseudo_gradient)]
        else:
            logger.warning("v_t or pseudo_gradient is None, skipping momentum update")

        # Bias correction - ensure m_t is not None
        if self.m_t is not None:
            m_hat = [m / (1 - self.beta_1**server_round) for m in self.m_t]
        else:
            logger.warning("m_t is None, cannot compute bias correction")
            return None, {}

        # Compute v_hat with proper None handling
        if self.v_t is not None:
            v_hat = [v / (1 - self.beta_2**server_round) for v in self.v_t]
        else:
            v_hat = [np.zeros_like(m) for m in m_hat]

        # Update weights - all variables are guaranteed non-None here
        new_weights = [
            w - self.eta * m / (np.sqrt(v) + self.tau)
            for w, m, v in zip(current_weights, m_hat, v_hat)
        ]

        self.current_parameters = ndarrays_to_parameters(new_weights)

        # --- Logging and Tracking ---
        final_metrics: Dict[str, Scalar] = {
            "variance": float(variance),
            "adapted_eta": float(self.eta),
            "avg_cosine_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "pseudo_gradient_norm": float(np.linalg.norm(flat_pseudo_gradient))
        }
        self.experiment_tracker.log_round_complete(server_round, final_metrics, results)

        return self.current_parameters, final_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and update adaptive loss kappas."""

        if not results:
            return None, {}

        # Aggregate loss using Flower's default
        try:
            loss_aggregated, agg_metrics = super().aggregate_evaluate(server_round, results, failures)
            metrics_aggregated = dict(agg_metrics) if agg_metrics else {}
        except (AttributeError, NotImplementedError):
            # Manual aggregation if super method doesn't exist
            loss_aggregated = 0.0
            metrics_aggregated: Dict[str, Scalar] = {}

        # --- Adaptive Loss Kappa Update ---
        # Collect Dice scores for each class from all clients
        class_dice_scores: List[List[float]] = [[] for _ in range(self.num_classes)]
        for _, res in results:
            for i in range(self.num_classes):
                score = res.metrics.get(f"dice_class_{i}", -1.0)
                if score != -1.0 and isinstance(score, (int, float)):
                    class_dice_scores[i].append(float(score))

        # Calculate average Dice for each class
        avg_class_dice = [np.mean(scores) if scores else 0.0 for scores in class_dice_scores]

        # Update kappa: kappa = lambda * (1 - Dice), higher loss -> higher kappa
        self.kappa_values = self.lambda_val * (1.0 - np.array(avg_class_dice))

        rounded_kappas = np.round(self.kappa_values, 3)
        logger.info(
            f"Round {server_round} Evaluation Complete. Updated Kappas: {rounded_kappas}"
        )
        metrics_aggregated["avg_fg_dice"] = float(
            np.mean(np.array(avg_class_dice[1:])))  # Exclude background

        return loss_aggregated, metrics_aggregated

    def _flatten_ndarrays(self, ndarrays: NDArrays) -> np.ndarray:
        """Flatten a list of numpy arrays into a single 1D array."""
        return np.concatenate([arr.flatten() for arr in ndarrays])

    def _calculate_update_variance(self,
                                   client_deltas: List[np.ndarray],
                                   pseudo_gradient: np.ndarray) -> Tuple[float, List[float]]:
        """Calculate variance of client updates using cosine similarity against the pseudo-gradient."""
        if not client_deltas or np.linalg.norm(pseudo_gradient) < self.tau:
            return 0.0, []

        similarities = []
        for delta in client_deltas:
            if np.linalg.norm(delta) > self.tau:
                sim = np.dot(delta, pseudo_gradient) / \
                    (np.linalg.norm(delta) * np.linalg.norm(pseudo_gradient))
                similarities.append(sim)

        return float(np.var(similarities)) if similarities else 0.0, similarities

    def _adapt_learning_rate(self, variance: float) -> float:
        """Adapt server learning rate based on variance."""
        return self.initial_eta / (1.0 + self.eta_adapt_rate * variance)

    def export_research_data(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export all research data, including FAUP and kappa values."""
        exported_files = self.experiment_tracker.export_research_data(output_dir)

        # Additional exports specific to this strategy
        if output_dir:
            export_dir = Path(output_dir)
        else:
            export_dir = Path(exported_files["server_metrics"]).parent

        # Export final client debts
        debt_data = [{"client_id": cid, "final_debt": debt}
                     for cid, debt in self.client_debts.items()]
        if debt_data:
            pd.DataFrame(debt_data).to_csv(export_dir / "faup_final_debts.csv", index=False)

        return exported_files 