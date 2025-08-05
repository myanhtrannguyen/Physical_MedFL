"""med: A Flower / PyTorch app for medical image segmentation."""

import sys
import os
from typing import Dict, Optional, Tuple

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from .strategy import AdaFedAdamStrategy
from .task import get_model, get_weights, get_testloader, test, set_weights
from .utils import export_and_plot_results, create_experiment_summary
from utils.metrics import evaluate_metrics
from collections import OrderedDict
import flwr as fl


class MedicalFLStrategy(AdaFedAdamStrategy):
    """Extended AdaFedAdamStrategy with post-processing capabilities."""
    
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
        """Handle post-processing after training completes with enhanced analysis."""
        try:
            print(f"Training completed after {self.rounds_completed} rounds")
            print("Exporting comprehensive research data...")
            
            # Export enhanced experiment data
            exported_files = self.experiment_tracker.export_research_data()
            print(f"Data exported: {list(exported_files.keys())}")
            
            # Get performance summary
            performance_summary = self.experiment_tracker.get_performance_summary()
            if performance_summary:
                print("EXPERIMENT SUMMARY:")
                print(f"  Final Accuracy: {performance_summary.get('final_accuracy', 0.0):.4f}")
                print(f"  Accuracy Improvement: {performance_summary.get('total_accuracy_improvement', 0.0):.4f}")
                print(f"  Convergence: {performance_summary.get('convergence_achieved', False)}")
                print(f"  Duration: {performance_summary.get('experiment_duration_minutes', 0.0):.1f} minutes")
                
                if 'final_fg_dice' in performance_summary:
                    print(f"  Final Dice: {performance_summary['final_fg_dice']:.4f}")
                    print(f"  Dice Improvement: {performance_summary.get('fg_dice_improvement', 0.0):.4f}")
            
            # Legacy post-processing
            export_and_plot_results(self, self.experiment_name)
            
            # Create enhanced experiment summary
            config = {
                "rounds_completed": self.rounds_completed,
                "strategy": "AdaFedAdamStrategy", 
                "experiment_name": self.experiment_name,
                "enhanced_tracking": True,
                "files_exported": list(exported_files.keys()),
                **performance_summary  # Include performance summary in config
            }
            output_dir = os.path.join("research_exports", self.experiment_name)
            create_experiment_summary(output_dir, self.experiment_name, config)
            
            print("Post-processing completed")
            
        except Exception as e:
            print(f"Error in post-processing: {e}")
            import traceback
            traceback.print_exc()


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
        
        # Evaluate model with comprehensive metrics
        loss, accuracy = test(net, testloader, device)
        
        # Get comprehensive metrics from utils
        metrics = evaluate_metrics(net, testloader, device, 4)  # N_CLASSES = 4
        fg_dice = sum(metrics['dice_scores'][1:]) / len(metrics['dice_scores'][1:]) if len(metrics['dice_scores']) > 1 else metrics['dice_scores'][0]
        fg_iou = sum(metrics['iou'][1:]) / len(metrics['iou'][1:]) if len(metrics['iou']) > 1 else metrics['iou'][0]
        fg_precision = sum(metrics['precision'][1:]) / len(metrics['precision'][1:]) if len(metrics['precision']) > 1 else metrics['precision'][0]
        fg_recall = sum(metrics['recall'][1:]) / len(metrics['recall'][1:]) if len(metrics['recall']) > 1 else metrics['recall'][0]
        fg_f1 = sum(metrics['f1_score'][1:]) / len(metrics['f1_score'][1:]) if len(metrics['f1_score']) > 1 else metrics['f1_score'][0]
        
        print(f"Server-side evaluation (Round {server_round}):")
        print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"  Dice: {fg_dice:.4f}, IoU: {fg_iou:.4f}")
        print(f"  Precision: {fg_precision:.4f}, Recall: {fg_recall:.4f}, F1: {fg_f1:.4f}")
        
        return loss, {
            "accuracy": accuracy,
            "centralized_accuracy": accuracy,
            "fg_dice": fg_dice,
            "fg_iou": fg_iou,
            "fg_precision": fg_precision,
            "fg_recall": fg_recall,
            "fg_f1": fg_f1
        }
    
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
        
        # AdaFedAdamStrategy specific parameters
        eta=0.0001,
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
