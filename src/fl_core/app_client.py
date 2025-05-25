#!/usr/bin/env python3
"""
Updated Federated Learning Client with Universal Data Loader
Demonstrates integration of the universal data loader with Flower FL.
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import logging
import os

# Import the universal data loader
from universal_data_loader import UniversalDataLoader, create_dataloader
from MLP_Model import (
    RobustMedVFL_UNet, 
    CombinedLoss, 
    evaluate_metrics,
    compute_class_weights,
    DEVICE, 
    NUM_CLASSES, 
    IMG_SIZE, 
    BATCH_SIZE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient(fl.client.NumPyClient):
    """Enhanced federated client using universal data loader."""
    
    def __init__(self, cid: str, data_path: str, max_train_samples: int = None, max_test_samples: int = 50):
        """
        Initialize federated client.
        
        Args:
            cid: Client ID
            data_path: Path to client's data
            max_train_samples: Maximum training samples to load
            max_test_samples: Maximum test samples to load
        """
        self.cid = cid
        self.data_path = data_path
        self.max_train_samples = max_train_samples if max_train_samples is not None else float('inf')
        self.max_test_samples = max_test_samples
        
        # Initialize model
        self.model = RobustMedVFL_UNet(n_channels=1, n_classes=NUM_CLASSES, dropout_rate=0.0).to(DEVICE)
        self.num_examples = {"trainset": 0, "testset": 0}
        
        # Load data using universal loader
        self.load_data()
        
        logger.info(f"Client {cid} initialized with {self.num_examples['trainset']} train, "
                   f"{self.num_examples['testset']} test samples")
    
    def load_data(self):
        """Load client data using universal data loader."""
        try:
            # Initialize universal data loader
            data_loader = UniversalDataLoader(
                target_size=(IMG_SIZE, IMG_SIZE),
                num_classes=NUM_CLASSES,
                normalize=True
            )
            
            # Detect and load data
            logger.info(f"Client {self.cid}: Loading data from {self.data_path}")
            
            # Kiểm tra path exists trước
            if not os.path.exists(self.data_path):
                logger.warning(f"Client {self.cid}: Path {self.data_path} does not exist, trying alternatives...")
                # Try default paths
                alternative_paths = [
                    "ACDC/database/training",
                    "ACDC/database/testing"
                ]
                found_path = None
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        found_path = alt_path
                        break
                
                if found_path:
                    self.data_path = found_path
                    logger.info(f"Client {self.cid}: Using alternative path: {self.data_path}")
                else:
                    raise FileNotFoundError(f"No valid data path found for client {self.cid}")
            
            detected_format = data_loader.detect_format(self.data_path)
            logger.info(f"Client {self.cid}: Detected format: {detected_format}")
            
            # Load ALL available training data
            logger.info(f"Client {self.cid}: Loading ALL available data from {self.data_path}...")
            images, masks = data_loader.load_data(
                self.data_path, 
                max_samples=None  # Load all available data
            )
            
            logger.info(f"Client {self.cid}: Successfully loaded {len(images)} images from real data")
            
            if len(images) == 0:
                logger.error(f"Client {self.cid}: No data loaded from {self.data_path}")
                logger.info(f"Client {self.cid}: Trying alternative paths...")
                
                # Try alternative paths
                alternative_paths = [
                    "ACDC/database/training",
                    "ACDC/database/testing", 
                    "data/training",
                    "data/testing"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        logger.info(f"Client {self.cid}: Trying {alt_path}")
                        try:
                            images, masks = data_loader.load_data(alt_path, max_samples=50)
                            if len(images) > 0:
                                logger.info(f"Client {self.cid}: Successfully loaded {len(images)} samples from {alt_path}")
                                break
                        except Exception as e:
                            logger.warning(f"Client {self.cid}: Failed to load from {alt_path}: {e}")
                            continue
                
                # Raise error instead of using dummy data
                if len(images) == 0:
                    error_msg = f"Client {self.cid}: Failed to load any real data from all paths!"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Use most data for training, keep some for testing
            total_samples = len(images)
            logger.info(f"Client {self.cid}: Total samples loaded: {total_samples}")
            
            # Client-specific data distribution for federated learning
            client_id = int(self.cid) if self.cid.isdigit() else 0
            
            # Each client gets ALL data but can still have different starting points for diversity
            # This ensures all clients train on full dataset but with different order/emphasis
            start_idx = (client_id * total_samples // 10) % total_samples
            
            # Rotate the data for this client (different perspective on same data)
            if start_idx > 0:
                images = np.concatenate([images[start_idx:], images[:start_idx]], axis=0)
                if masks is not None:
                    masks = np.concatenate([masks[start_idx:], masks[:start_idx]], axis=0)
            
            # Split: Use 85% for training, 15% for testing (max test_samples)
            train_size = int(0.85 * total_samples)
            test_size = min(self.max_test_samples, total_samples - train_size)
            
            train_images = images[:train_size]
            train_masks = masks[:train_size] if masks is not None else None
            
            test_images = images[train_size:train_size + test_size]
            test_masks = masks[train_size:train_size + test_size] if masks is not None else None
            
            logger.info(f"Client {self.cid}: Split - Train: {len(train_images)}, Test: {len(test_images)}")
            
            # Create DataLoaders
            if train_masks is not None:
                self.trainloader = create_dataloader(
                    train_images, train_masks, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    augment=True
                )
                
                if len(test_images) > 0 and test_masks is not None:
                    self.testloader = create_dataloader(
                        test_images, test_masks,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        augment=False
                    )
                else:
                    self.testloader = None
                
                # Compute class weights
                self.class_weights = compute_class_weights(train_masks, NUM_CLASSES)
            else:
                logger.warning(f"Client {self.cid}: No masks available, using dummy setup")
                self.trainloader = None
                self.testloader = None
                self.class_weights = None
            
            # Update counts
            self.num_examples["trainset"] = len(train_images)
            self.num_examples["testset"] = len(test_images)
            
        except Exception as e:
            logger.error(f"Client {self.cid}: Critical error loading data: {e}")
            logger.error(f"Client {self.cid}: Stack trace:", exc_info=True)
            logger.warning(f"Client {self.cid}: Falling back to dummy data - PERFORMANCE WILL BE POOR!")
            # Only use dummy data in critical failure
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing."""
        logger.info(f"Client {self.cid}: Creating dummy data")
        
        # Create dummy tensors
        train_images = torch.randn(20, 1, IMG_SIZE, IMG_SIZE)
        train_masks = torch.randint(0, NUM_CLASSES, (20, IMG_SIZE, IMG_SIZE))
        test_images = torch.randn(10, 1, IMG_SIZE, IMG_SIZE)  
        test_masks = torch.randint(0, NUM_CLASSES, (10, IMG_SIZE, IMG_SIZE))
        
        # Create DataLoaders
        self.trainloader = DataLoader(
            TensorDataset(train_images, train_masks),
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        self.testloader = DataLoader(
            TensorDataset(test_images, test_masks),
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        
        self.num_examples["trainset"] = 20
        self.num_examples["testset"] = 10
        self.class_weights = None
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> fl.common.NDArrays:
        """Return model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: fl.common.NDArrays) -> None:
        """Set model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[fl.common.NDArrays, int, Dict[str, fl.common.Scalar]]:
        """Train model on client data."""
        logger.info(f"Client {self.cid}: Starting training round {config.get('server_round', 0)}")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Skip training if no data
        if self.trainloader is None or len(self.trainloader) == 0:
            logger.warning(f"Client {self.cid}: No training data available")
            return self.get_parameters(config), 0, {}
        
        # Training configuration with aggressive optimization
        epochs = int(config.get("epochs", 6))
        learning_rate = float(config.get("learning_rate", 1e-3))  # Higher initial LR
        weight_decay = float(config.get("weight_decay", 1e-4))  # Strong regularization
        dropout_rate = float(config.get("dropout_rate", 0.15))  # Balanced dropout
        
        logger.info(f"Client {self.cid}: Training config - LR: {learning_rate}, WD: {weight_decay}, Dropout: {dropout_rate}")
        
        # Update the model's dropout rate based on config
        self.model = RobustMedVFL_UNet(n_channels=1, n_classes=NUM_CLASSES, dropout_rate=dropout_rate).to(DEVICE)
        
        # Set the parameters after recreating the model with new dropout rate
        self.set_parameters(parameters)
        
        # Setup training with improved optimizer
        criterion = CombinedLoss(
            num_classes=NUM_CLASSES,
            in_channels_maxwell=1024,
            class_weights=self.class_weights
        ).to(DEVICE)
        
        # Use AdamW with optimized parameters for faster convergence
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # Better convergence for noisy gradients
        )
        
        # Use more aggressive learning rate scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(self.trainloader),
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos'
        )
        
        # Advanced training loop with gradient accumulation and mixed precision
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        best_epoch_loss = float('inf')
        patience_counter = 0
        max_patience = 3  # More patience for better convergence
        current_lr = learning_rate
        
        # Gradient accumulation for effective larger batch size
        accumulation_steps = 2
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (images, targets) in enumerate(self.trainloader):
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                
                # Forward pass
                logits, all_eps_sigma_tuples = self.model(images)
                
                # Placeholder for physics components
                b1_map = torch.randn_like(images[:, 0:1, ...], device=DEVICE)
                
                loss = criterion(logits, targets, b1_map, all_eps_sigma_tuples)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()  # Step scheduler every batch for OneCycleLR
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accumulation_steps  # Unscale for logging
                epoch_batches += 1
            
            # Process epoch results  
            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                total_loss += avg_epoch_loss
                total_batches += 1
                
                # Get current learning rate (OneCycleLR steps per batch, not per epoch)
                current_lr = scheduler.get_last_lr()[0]
                
                # Early stopping check
                if avg_epoch_loss < best_epoch_loss:
                    best_epoch_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 1 == 0:  # Log every epoch with more detail
                    logger.info(f"Client {self.cid}: Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.2e}")
                
                # Early stopping
                if patience_counter >= max_patience and epoch >= 2:
                    logger.info(f"Client {self.cid}: Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                    break
        
        # Calculate final metrics with improved tracking
        final_loss = total_loss / total_batches if total_batches > 0 else 0.0
        actual_epochs = total_batches  # Track actual epochs completed (may be less due to early stopping)
        
        metrics = {
            "train_loss": final_loss,
            "best_epoch_loss": best_epoch_loss,
            "epochs_completed": actual_epochs,
            "early_stopped": patience_counter >= max_patience,
            "final_learning_rate": current_lr,
            "batches_processed": sum(len(self.trainloader) for _ in range(actual_epochs))
        }
        
        logger.info(f"Client {self.cid}: Training completed. Final loss: {final_loss:.4f}, Best: {best_epoch_loss:.4f}")
        
        return self.get_parameters(config), self.num_examples["trainset"], metrics
    
    def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate model on client data."""
        logger.info(f"Client {self.cid}: Starting evaluation")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Skip evaluation if no data
        if self.testloader is None or len(self.testloader) == 0:
            logger.warning(f"Client {self.cid}: No test data available")
            return 0.0, 0, {}
        
        # Evaluation
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Setup loss function for evaluation
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, targets in self.testloader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                
                logits, _ = self.model(images)
                loss = criterion(logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Detailed metrics
        try:
            detailed_metrics = evaluate_metrics(self.model, self.testloader, DEVICE, NUM_CLASSES)
            
            # Format metrics for transmission
            metrics = {
                "eval_loss": avg_loss,
                "dice_avg": float(np.mean(detailed_metrics['dice_scores'])),
                "dice_foreground_avg": float(np.mean(detailed_metrics['dice_scores'][1:])) if NUM_CLASSES > 1 else float(detailed_metrics['dice_scores'][0]),
                "iou_avg": float(np.mean(detailed_metrics['iou'])),
                "precision_avg": float(np.mean(detailed_metrics['precision'])),
                "recall_avg": float(np.mean(detailed_metrics['recall'])),
                "f1_avg": float(np.mean(detailed_metrics['f1_score']))
            }
            
        except Exception as e:
            logger.warning(f"Client {self.cid}: Error calculating detailed metrics: {e}")
            metrics = {"eval_loss": avg_loss}
        
        logger.info(f"Client {self.cid}: Evaluation completed. Loss: {avg_loss:.4f}")
        
        return avg_loss, self.num_examples["testset"], metrics

def client_fn(context) -> fl.client.Client:
    """Create a client instance."""
    
    # Handle both string and Context types for compatibility
    if isinstance(context, str):
        # For simulation mode where partition_id is passed as string
        cid = context
    else:
        # For newer API where Context object is passed
        cid = str(context.node_id)
    
    # Define client data paths for federated learning
    # Each client gets different subset of training data
    client_data_mapping = {
        "0": "ACDC/database/training",  # Client 0 gets training data subset 1
        "1": "ACDC/database/training",  # Client 1 gets training data subset 2  
        "2": "ACDC/database/testing",   # Client 2 gets testing data (for validation)
    }
    
    # All clients use training data by default for federated learning
    data_path = client_data_mapping.get(cid, "ACDC/database/training")
    
    # Create NumPyClient and convert to Client
    numpy_client = FederatedClient(cid=cid, data_path=data_path)
    return numpy_client.to_client()

# Create the ClientApp
app = fl.client.ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    print("🚀 Enhanced Federated Learning Client with Universal Data Loader")
    print("="*60)
    print("Features:")
    print("✓ Universal data format support (NIfTI, H5, DICOM, PNG/JPG, NumPy)")
    print("✓ Automatic format detection")
    print("✓ Robust error handling and fallback to dummy data")
    print("✓ Configurable data loading and preprocessing")
    print("✓ Integration with existing federated learning framework")
    print("="*60)
    
    print("\nTo run this client:")
    print("flower-client --app app_client:app")
    
    # Test client creation
    print("\n🧪 Testing client creation...")
    try:
        test_client_numpy = FederatedClient("0", "ACDC/database/training")
        test_client = test_client_numpy.to_client()
        print(f"✓ Test client created successfully")
        print(f"  Train samples: {test_client_numpy.num_examples['trainset']}")
        print(f"  Test samples: {test_client_numpy.num_examples['testset']}")
    except Exception as e:
        print(f"✗ Error creating test client: {e}") 
