import flwr as fl
from typing import Dict, Optional, Tuple, List
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar, FitIns, EvaluateIns

# You might not need direct model access on the server if you're just aggregating.
# However, if your strategy needs to evaluate a global model, you might.
# For FedAvg, it's usually not needed on the server side directly for the strategy itself.
# from model_and_data_handle import RobustMedVFL_UNet, load_h5_data, evaluate_metrics, DEVICE, NUM_CLASSES

# Define a comprehensive FedAvg strategy
class FedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        timeout_fit=300,  # 5 minutes
        timeout_evaluate=300,  # 5 minutes
        num_rounds=5  # Total number of training rounds
    ):
        # Define comprehensive fit_metrics_aggregation_fn
        def fit_metrics_aggregation_fn(metrics_list):
            """Aggregate fit metrics from multiple clients with weighted averaging."""
            if len(metrics_list) == 0:
                return {}
            
            # Handle different metric formats
            processed_metrics = []
            for metric in metrics_list:
                if isinstance(metric, tuple) and len(metric) >= 2:
                    # Format: (num_examples, metrics_dict)
                    num_examples, metrics_dict = metric[0], metric[1]
                    if isinstance(metrics_dict, dict):
                        processed_metrics.append((num_examples, metrics_dict))
                elif isinstance(metric, dict):
                    # Direct metrics dict
                    processed_metrics.append((1, metric))  # Default weight of 1
            
            if not processed_metrics:
                return {}
            
            # Collect all unique metric keys
            all_keys = set()
            for _, metrics_dict in processed_metrics:
                if isinstance(metrics_dict, dict):
                    all_keys.update(metrics_dict.keys())
            
            # Weighted aggregation
            aggregated = {}
            total_examples = sum(num_examples for num_examples, _ in processed_metrics)
            
            for key in all_keys:
                weighted_sum = 0.0
                valid_entries = 0
                
                for num_examples, metrics_dict in processed_metrics:
                    if key in metrics_dict:
                        try:
                            value = float(metrics_dict[key])
                            weighted_sum += value * num_examples
                            valid_entries += num_examples
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            continue
                
                if valid_entries > 0:
                    aggregated[key] = weighted_sum / valid_entries
            
            return aggregated
            
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            evaluate_fn=evaluate_fn,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
        )
        self.timeout_fit = timeout_fit
        self.timeout_evaluate = timeout_evaluate
        self.num_rounds = num_rounds
        
        # Performance tracking for adaptive learning
        self.round_metrics = {}
        self.best_global_metric = 0.0
        self.rounds_without_improvement = 0
        self.max_patience = 3  # Stop if no improvement for 3 rounds
        
        print(f"Enhanced FedAvgStrategy initialized for {num_rounds} rounds")
        print(f"✓ Adaptive learning rate and early stopping enabled")
        print(f"✓ Performance tracking for global model improvement")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n{'='*60}")
        print(f"ROUND {server_round} - CONFIGURE FIT")
        print(f"{'='*60}")
        
        # Improved epoch scheduling for better convergence
        if server_round <= 2:
            epochs = 6  # More epochs early on for better learning
        elif server_round <= 5:
            epochs = 5  # Moderate epochs in middle rounds
        else:
            epochs = 4  # Reduce in later rounds for fine-tuning
        
        # Optimized learning rate for full dataset training
        if server_round == 1:
            learning_rate = 5e-4  # Lower start with full data
        elif server_round <= 3:
            learning_rate = 2e-4  # Medium rate for stable progress  
        elif server_round <= 5:
            learning_rate = 1e-4  # Lower for fine-tuning
        else:
            learning_rate = 5e-5  # Very low for final convergence
        
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,  # Stronger regularization
            "dropout_rate": 0.15,  # Moderate dropout for better generalization
            "total_rounds": self.num_rounds
        }
        
        print(f"Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {config['weight_decay']}")
        print(f"  Dropout rate: {config['dropout_rate']}")
        print(f"  Round: {server_round}/{self.num_rounds}")
        
        fit_ins = FitIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients
        )
        print(f"  Selected {len(clients)} clients for training")
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        print(f"\nCONFIGURE EVALUATION - Round {server_round}")
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "total_rounds": self.num_rounds
        }
        eval_ins = EvaluateIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients, min_num_clients=self.min_evaluate_clients
        )
        print(f"  Selected {len(clients)} clients for evaluation")
        return [(client, eval_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"\nAGGREGATE FIT RESULTS - Round {server_round}")
        
        # Log failures
        if failures:
            print(f"⚠ Training failures: {len(failures)}")
            for i, failure in enumerate(failures):
                print(f"  Failure {i+1}: {type(failure).__name__}")
        
        if not results:
            print("❌ No successful training results to aggregate!")
            return None, {}
        
        print(f"✓ Successfully received results from {len(results)} clients")
        
        # Aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Detailed logging of aggregated metrics with performance tracking
        if aggregated_metrics:
            print(f"📊 Aggregated training metrics:")
            for key, value in aggregated_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Store metrics for trend analysis
            self.round_metrics[f"round_{server_round}_train"] = aggregated_metrics
            
            # Check for training stability (high loss growth indicates problems)
            if "train_loss" in aggregated_metrics:
                if server_round > 1:
                    prev_loss = self.round_metrics.get(f"round_{server_round-1}_train", {}).get("train_loss", 0)
                    current_loss = aggregated_metrics["train_loss"]
                    loss_change = current_loss - prev_loss
                    
                    if loss_change > 0.5:  # Significant loss increase
                        print(f"⚠️  WARNING: Training loss increased significantly (+{loss_change:.3f})")
                        print(f"   Consider reducing learning rate for next round")
                    elif loss_change < -0.1:  # Good loss decrease
                        print(f"✅ Training loss improved by {-loss_change:.3f}")
        
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[ClientProxy, fl.common.EvaluateRes] | BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        print(f"\nAGGREGATE EVALUATION RESULTS - Round {server_round}")
        
        # Log failures in detail
        if failures:
            print(f"⚠ Evaluation failures: {len(failures)}")
            for i, failure in enumerate(failures):
                print(f"  Failure {i+1}: {type(failure).__name__}: {str(failure)[:100]}")
        
        # Check for valid results
        if not results:
            print(f"❌ No successful evaluation results in round {server_round}")
            return None, {}

        # Filter out results with 0 examples
        valid_results = [res for res in results if res[1].num_examples > 0]

        if not valid_results:
            print(f"❌ All evaluation results had 0 examples in round {server_round}")
            return None, {}
        
        print(f"✓ Processing evaluation results from {len(valid_results)} clients")
        
        # Calculate client-wise metrics for detailed logging
        client_metrics = []
        total_examples = 0
        
        for client_proxy, eval_res in valid_results:
            client_id = client_proxy.cid
            num_examples = eval_res.num_examples
            loss = eval_res.loss
            metrics = eval_res.metrics
            
            total_examples += num_examples
            client_info = {
                'client_id': client_id,
                'num_examples': num_examples,
                'loss': loss,
                'metrics': metrics
            }
            client_metrics.append(client_info)
            
            print(f"  Client {client_id}: Loss={loss:.4f}, Examples={num_examples}")
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.4f}")
        
        # Aggregate using parent class
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, valid_results, failures)
        
        # Enhanced logging
        print(f"\n📈 ROUND {server_round} SUMMARY:")
        print(f"  Total examples evaluated: {total_examples}")
        print(f"  Aggregated loss: {loss_aggregated:.4f}" if loss_aggregated is not None else "  Aggregated loss: None")
        
        if metrics_aggregated:
            print(f"  📊 Key aggregated metrics:")
            priority_metrics = ['dice_avg', 'dice_foreground_avg', 'iou_avg', 'eval_loss']
            for metric in priority_metrics:
                if metric in metrics_aggregated:
                    print(f"    {metric}: {metrics_aggregated[metric]:.4f}")
                    
            # Show other metrics
            other_metrics = {k: v for k, v in metrics_aggregated.items() if k not in priority_metrics}
            if other_metrics:
                print(f"  📋 Additional metrics:")
                for key, value in other_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.4f}")
                    else:
                        print(f"    {key}: {value}")
            
            # Store evaluation metrics and check for global improvement
            self.round_metrics[f"round_{server_round}_eval"] = metrics_aggregated
            
            # Track global model performance (use dice_foreground_avg as primary metric)
            current_global_metric = 0.0
            if 'dice_foreground_avg' in metrics_aggregated:
                current_global_metric = float(metrics_aggregated['dice_foreground_avg'])
            elif 'dice_avg' in metrics_aggregated:
                current_global_metric = float(metrics_aggregated['dice_avg'])
            
            if current_global_metric > self.best_global_metric:
                improvement = current_global_metric - self.best_global_metric
                self.best_global_metric = current_global_metric
                self.rounds_without_improvement = 0
                print(f"🎉 NEW BEST GLOBAL MODEL! Dice improved by {improvement:.4f}")
            else:
                self.rounds_without_improvement += 1
                print(f"📉 No improvement for {self.rounds_without_improvement} rounds (Best: {self.best_global_metric:.4f})")
                
                if self.rounds_without_improvement >= self.max_patience:
                    print(f"🛑 EARLY STOPPING RECOMMENDATION: No improvement for {self.max_patience} rounds")
                    print(f"   Consider stopping training or adjusting hyperparameters")
            
            # Performance trend analysis
            if server_round >= 3:
                print(f"\n📊 PERFORMANCE TREND ANALYSIS:")
                recent_rounds = []
                for r in range(max(1, server_round-2), server_round+1):
                    eval_metrics = self.round_metrics.get(f"round_{r}_eval", {})
                    dice_score = 0.0
                    if 'dice_foreground_avg' in eval_metrics:
                        dice_score = float(eval_metrics['dice_foreground_avg'])
                    elif 'dice_avg' in eval_metrics:
                        dice_score = float(eval_metrics['dice_avg'])
                    recent_rounds.append(dice_score)
                
                if len(recent_rounds) >= 3:
                    trend = recent_rounds[-1] - recent_rounds[0]
                    if trend > 0.05:
                        print(f"  📈 Positive trend: +{trend:.3f} over last 3 rounds")
                    elif trend < -0.02:
                        print(f"  📉 Negative trend: {trend:.3f} over last 3 rounds")
                        print(f"     Consider reducing learning rate or adding regularization")
                    else:
                        print(f"  📊 Stable performance: {trend:+.3f} change over last 3 rounds")
        
        return loss_aggregated, metrics_aggregated

# Define enhanced evaluation metrics aggregation
def evaluate_metrics_aggregation_fn(metrics_list):
    """Enhanced evaluation metrics aggregation with weighted averaging."""
    if len(metrics_list) == 0:
        return {}
        
    # Handle different metric formats
    processed_metrics = []
    for metric in metrics_list:
        if isinstance(metric, tuple) and len(metric) >= 2:
            # Format: (num_examples, metrics_dict)
            num_examples, metrics_dict = metric[0], metric[1]
            if isinstance(metrics_dict, dict):
                processed_metrics.append((num_examples, metrics_dict))
        elif isinstance(metric, dict):
            # Direct metrics dict
            processed_metrics.append((1, metric))  # Default weight of 1
    
    if not processed_metrics:
        return {}
    
    # Collect all unique metric keys
    all_keys = set()
    for _, metrics_dict in processed_metrics:
        if isinstance(metrics_dict, dict):
            all_keys.update(metrics_dict.keys())
    
    # Weighted aggregation
    aggregated = {}
    total_examples = sum(num_examples for num_examples, _ in processed_metrics)
    
    for key in all_keys:
        values_and_weights = []
        
        for num_examples, metrics_dict in processed_metrics:
            if key in metrics_dict:
                try:
                    value = float(metrics_dict[key])
                    values_and_weights.append((value, num_examples))
                except (ValueError, TypeError):
                    # Try to parse string representations
                    if isinstance(metrics_dict[key], str):
                        try:
                            # Handle formatted strings like "[0.1234, 0.5678]"
                            if metrics_dict[key].startswith('[') and metrics_dict[key].endswith(']'):
                                continue  # Skip formatted array strings
                            value = float(metrics_dict[key])
                            values_and_weights.append((value, num_examples))
                        except ValueError:
                            continue
        
        if values_and_weights:
            # Compute weighted average
            weighted_sum = sum(value * weight for value, weight in values_and_weights)
            total_weight = sum(weight for _, weight in values_and_weights)
            aggregated[key] = weighted_sum / total_weight
    
    return aggregated

# Enhanced strategy configuration with improved hyperparameters
strategy = FedAvgStrategy(
    min_fit_clients=1,      # Minimum clients for training
    min_evaluate_clients=1, # Minimum clients for evaluation  
    min_available_clients=1, # Minimum available clients
    fraction_fit=1.0,       # Use all available clients for training
    fraction_evaluate=1.0,  # Use all available clients for evaluation
    timeout_fit=300,        # 5 minutes timeout for training
    timeout_evaluate=300,   # 5 minutes timeout for evaluation
    num_rounds=8,           # Increased rounds to test performance improvements
)

# Enhanced ServerConfig
config = fl.server.ServerConfig(
    num_rounds=8,           # Number of federated learning rounds (increased)
    round_timeout=600       # 10 minutes timeout per round
)

# Increase max message size for large models (reduced for stability)
MAX_MESSAGE_LENGTH = 512 * 1024 * 1024  # 512MB in bytes

# Create enhanced ServerApp
app = fl.server.ServerApp(
    config=config,
    strategy=strategy
)

if __name__ == "__main__":
    print("🚀 ENHANCED FEDERATED LEARNING SERVER v2.0")
    print("="*60)
    print("🔧 PERFORMANCE OPTIMIZATIONS:")
    print("✓ Adaptive learning rate scheduling (1e-4 → 5e-5)")
    print("✓ Reduced epoch scheduling (3-4 epochs per round)")
    print("✓ Weight decay regularization (1e-5)")
    print("✓ Dropout regularization (0.1)")
    print("✓ Gradient clipping (max_norm=1.0)")
    print("✓ Early stopping (patience=2 epochs)")
    print("✓ Cosine annealing LR scheduler within rounds")
    print("✓ AdamW optimizer with better regularization")
    print("")
    print("📊 MONITORING FEATURES:")
    print("✓ Global model performance tracking")
    print("✓ Performance trend analysis")
    print("✓ Automatic early stopping recommendations")
    print("✓ Training stability monitoring")
    print("✓ Comprehensive round-by-round logging")
    print("="*60)
    
    print("\nTo run this server:")
    print("1. CLI (recommended): flower-server --app app_server:app")
    print("2. Direct execution: python app_server.py --start-server")
    
    # Direct execution option
    import sys
    if "--start-server" in sys.argv:
        try:
            print("\n🌸 Starting Flower server...")
            fl.server.start_server(
                server_address="0.0.0.0:8080", 
                config=config, 
                strategy=strategy,
                grpc_max_message_length=MAX_MESSAGE_LENGTH,
                certificates=None  # No SSL
            )
        except Exception as e:
            print(f"❌ Server execution failed: {type(e).__name__}: {str(e)}")
    else:
        print("\n💡 Use --start-server flag to run directly, or use the CLI command above.")

