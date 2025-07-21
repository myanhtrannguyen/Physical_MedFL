"""med: A Flower / PyTorch app for medical image segmentation."""

import sys
import os
from typing import Dict, Optional, Tuple

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from .strategy import UnifiedFairnessStrategy
from .task import get_model, get_weights, get_testloader, test, set_weights
from .utils import export_and_plot_results, create_experiment_summary
from collections import OrderedDict
import flwr as fl


class MedicalFLStrategy(UnifiedFairnessStrategy):
    """Extended UnifiedFairnessStrategy with post-processing capabilities."""
    
    def __init__(self, experiment_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name
        self.rounds_completed = 0
        self.max_rounds = 0
        
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Override to track max rounds."""
        self.max_rounds = max(self.max_rounds, server_round)
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Override to handle post-processing after final round."""
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        self.rounds_completed = server_round
        
        # If this is the final round, trigger post-processing
        if server_round >= self.max_rounds:
            self._post_process_results()
            
        return aggregated_result
    
    def _post_process_results(self):
        """Handle post-processing after training completes."""
        try:
            print(f"\n=== Training completed after {self.rounds_completed} rounds ===")
            export_and_plot_results(self, self.experiment_name)
            
            # Create experiment summary
            config = {
                "rounds_completed": self.rounds_completed,
                "strategy": "UnifiedFairnessStrategy", 
                "experiment_name": self.experiment_name
            }
            output_dir = os.path.join("research_exports", self.experiment_name)
            create_experiment_summary(output_dir, self.experiment_name, config)
            
        except Exception as e:
            print(f"Error in post-processing: {e}")


def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    testloader = get_testloader()
    
    if testloader is None:
        return None
        
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate global model on the test set."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = get_model().to(device)
        
        # Set model parameters
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        
        # Evaluate model
        loss, accuracy = test(net, testloader, device)
        
        print(f"Server-side evaluation (Round {server_round}): Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        return loss, {"accuracy": accuracy}
    
    return evaluate


def server_fn(context: Context):
    """Create and return server components."""
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = float(context.run_config["fraction-fit"])
    
    # Get experiment config
    experiment_name = str(context.run_config.get("experiment-name", "ACDC_Medical_FL"))
    min_fit_clients = int(context.run_config.get("min-fit-clients", 2))
    min_available_clients = int(context.run_config.get("min-available-clients", 10))
    
    # Initialize model parameters
    net = get_model()
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy - Use extended MedicalFLStrategy
    strategy = MedicalFLStrategy(
        experiment_name=experiment_name,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=max(2, int(min_available_clients * 0.3)),
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(),
        
        # UnifiedFairnessStrategy specific parameters
        eta=0.01,
        eta_adapt_rate=1.5,
        w_impact=0.4,
        w_debt=0.6,
        num_classes=4,
        lambda_val=15.0
    )
    
    # Set max rounds for post-processing
    strategy.max_rounds = num_rounds
    
    config = ServerConfig(num_rounds=num_rounds)

    print(f"Starting Federated Learning Simulation: {experiment_name}")
    print(f"Server rounds: {num_rounds}, Fraction fit: {fraction_fit}")

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
