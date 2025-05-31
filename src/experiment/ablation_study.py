"""
Comprehensive Ablation Study Framework for High-Tier Conference Publication
Analyzes contribution of each component in RobustMedVFL_UNet
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings

from src.models.mlp_model import (
    RobustMedVFL_UNet, 
    MaxwellSolver,
    ePURE,
    quantum_noise_injection,
    adaptive_spline_smoothing,
    CombinedLoss,
    DiceLoss,
    PhysicsLoss
)
from src.data.research_loader import create_research_dataloader
from src.utils.seed import set_seed
from src.utils.logger import setup_federated_logger

warnings.filterwarnings('ignore')

class AblationStudy:
    """
    Comprehensive ablation study framework for medical image segmentation.
    
    Components to ablate:
    1. Maxwell Solver (Physics Constraints)
    2. ePURE (Noise Estimation)
    3. Quantum Noise Injection
    4. Adaptive Spline Smoothing
    5. Combined Loss vs Standard Loss
    6. Different loss weights
    
    Analysis:
    - Component contribution analysis
    - Component interaction effects
    - Sensitivity analysis
    - Computational overhead analysis
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 output_dir: str = "results/ablation_study"):
        
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_federated_logger(
            client_id="ablation_study",
            log_dir=str(self.output_dir / "logs")
        )
        
        # Component configurations
        self.components = {
            'maxwell_solver': {
                'description': 'Physics-informed Maxwell solver constraints',
                'enabled': True,
                'weight': 0.1
            },
            'epure': {
                'description': 'ePURE noise estimation and correction',
                'enabled': True,
                'weight': 1.0
            },
            'quantum_noise': {
                'description': 'Quantum noise injection for robustness',
                'enabled': True,
                'factor': 0.01
            },
            'adaptive_smoothing': {
                'description': 'Adaptive spline smoothing preprocessing',
                'enabled': True,
                'kernel_size': 5
            },
            'combined_loss': {
                'description': 'Combined loss (CE + Dice + Physics + Smoothness)',
                'enabled': True,
                'weights': {'ce': 0.5, 'dice': 0.5, 'physics': 0.1, 'smoothness': 0.01}
            }
        }
        
        # Results storage
        self.results = {}
        self.component_contributions = {}
        
        self.logger.info(f"Ablation study initialized for {len(self.components)} components")

    def create_ablated_model(self, ablated_components: List[str]) -> nn.Module:
        """
        Create model with specified components ablated (removed).
        
        Args:
            ablated_components: List of component names to remove
            
        Returns:
            Model with specified components disabled
        """
        
        class AblatedRobustMedVFL_UNet(nn.Module):
            def __init__(self, config, ablated_components, device):
                super().__init__()
                self.config = config
                self.ablated_components = ablated_components
                self.device = device
                
                # Basic U-Net architecture (always present)
                self.enc1 = self._create_encoder_block(1, 64, 'enc1')
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = self._create_encoder_block(64, 128, 'enc2')
                self.pool2 = nn.MaxPool2d(2)
                self.enc3 = self._create_encoder_block(128, 256, 'enc3')
                self.pool3 = nn.MaxPool2d(2)
                self.enc4 = self._create_encoder_block(256, 512, 'enc4')
                self.pool4 = nn.MaxPool2d(2)
                
                # Bottleneck
                self.bottleneck = self._create_encoder_block(512, 1024, 'bottleneck')
                
                # Decoder
                self.dec1 = self._create_decoder_block(1024, 512, 512, 'dec1')
                self.dec2 = self._create_decoder_block(512, 256, 256, 'dec2')
                self.dec3 = self._create_decoder_block(256, 128, 128, 'dec3')
                self.dec4 = self._create_decoder_block(128, 64, 64, 'dec4')
                
                # Output
                self.out_conv = nn.Conv2d(64, config.get('n_classes', 4), kernel_size=1)
                
                # Optional components
                self.maxwell_enabled = 'maxwell_solver' not in ablated_components
                self.epure_enabled = 'epure' not in ablated_components
                self.smoothing_enabled = 'adaptive_smoothing' not in ablated_components
                
                if self.maxwell_enabled:
                    self.maxwell_solvers = nn.ModuleList([
                        MaxwellSolver(in_channels=1024),
                        MaxwellSolver(in_channels=512),
                        MaxwellSolver(in_channels=256),
                        MaxwellSolver(in_channels=128)
                    ])
                
                if self.epure_enabled:
                    self.epure_modules = nn.ModuleList([
                        ePURE(in_channels=1),
                        ePURE(in_channels=64),
                        ePURE(in_channels=128),
                        ePURE(in_channels=256),
                        ePURE(in_channels=512)
                    ])
            
            def _create_encoder_block(self, in_channels, out_channels, name):
                """Create encoder block with optional components"""
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def _create_decoder_block(self, in_channels, skip_channels, out_channels, name):
                """Create decoder block with optional components"""
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                concat_ch = in_channels // 2 + skip_channels
                
                conv_block = nn.Sequential(
                    nn.Conv2d(concat_ch, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                
                return conv_block
            
            def forward(self, x):
                # Apply preprocessing components
                if self.smoothing_enabled and self.epure_enabled and hasattr(self, 'epure_modules'):
                    noise_profile = self.epure_modules[0](x)
                    x = adaptive_spline_smoothing(x, noise_profile)
                
                # Encoder
                e1 = self.enc1(x)
                p1 = self.pool1(e1)
                
                e2 = self.enc2(p1)
                p2 = self.pool2(e2)
                
                e3 = self.enc3(p2)
                p3 = self.pool3(e3)
                
                e4 = self.enc4(p3)
                p4 = self.pool4(e4)
                
                # Bottleneck
                b = self.bottleneck(p4)
                
                # Decoder with optional physics constraints
                physics_outputs = []
                
                # Decoder 1
                d1_up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2).to(self.device)(b)
                diffY = e4.size()[2] - d1_up.size()[2]
                diffX = e4.size()[3] - d1_up.size()[3]
                d1_up = nn.functional.pad(d1_up, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
                d1_cat = torch.cat([e4, d1_up], dim=1)
                
                if self.maxwell_enabled and hasattr(self, 'maxwell_solvers'):
                    eps, sigma = self.maxwell_solvers[0](d1_cat)
                    physics_outputs.append((eps, sigma))
                
                d1 = self.dec1(d1_cat)
                
                # Decoder 2
                d2_up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2).to(self.device)(d1)
                diffY = e3.size()[2] - d2_up.size()[2]
                diffX = e3.size()[3] - d2_up.size()[3]
                d2_up = nn.functional.pad(d2_up, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
                d2_cat = torch.cat([e3, d2_up], dim=1)
                
                if self.maxwell_enabled and hasattr(self, 'maxwell_solvers'):
                    eps, sigma = self.maxwell_solvers[1](d2_cat)
                    physics_outputs.append((eps, sigma))
                
                d2 = self.dec2(d2_cat)
                
                # Decoder 3
                d3_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2).to(self.device)(d2)
                diffY = e2.size()[2] - d3_up.size()[2]
                diffX = e2.size()[3] - d3_up.size()[3]
                d3_up = nn.functional.pad(d3_up, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
                d3_cat = torch.cat([e2, d3_up], dim=1)
                
                if self.maxwell_enabled and hasattr(self, 'maxwell_solvers'):
                    eps, sigma = self.maxwell_solvers[2](d3_cat)
                    physics_outputs.append((eps, sigma))
                
                d3 = self.dec3(d3_cat)
                
                # Decoder 4
                d4_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2).to(self.device)(d3)
                diffY = e1.size()[2] - d4_up.size()[2]
                diffX = e1.size()[3] - d4_up.size()[3]
                d4_up = nn.functional.pad(d4_up, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
                d4_cat = torch.cat([e1, d4_up], dim=1)
                
                if self.maxwell_enabled and hasattr(self, 'maxwell_solvers'):
                    eps, sigma = self.maxwell_solvers[3](d4_cat)
                    physics_outputs.append((eps, sigma))
                
                d4 = self.dec4(d4_cat)
                
                # Output
                output = self.out_conv(d4)
                
                if self.maxwell_enabled and physics_outputs:
                    return output, physics_outputs
                else:
                    return output
        
        return AblatedRobustMedVFL_UNet(self.config, ablated_components, self.device).to(self.device)

    def create_ablated_loss(self, ablated_components: List[str]) -> nn.Module:
        """
        Create loss function with specified components ablated.
        
        Args:
            ablated_components: List of component names to remove
            
        Returns:
            Loss function with specified components disabled
        """
        
        if 'combined_loss' in ablated_components:
            # Use simple CrossEntropy loss
            return nn.CrossEntropyLoss()
        
        class AblatedCombinedLoss(nn.Module):
            def __init__(self, ablated_components, config):
                super().__init__()
                self.ablated_components = ablated_components
                self.config = config
                
                # Always include basic losses
                self.ce_loss = nn.CrossEntropyLoss()
                self.dice_loss = DiceLoss(num_classes=config.get('n_classes', 4))
                
                # Optional physics loss
                if 'maxwell_solver' not in ablated_components:
                    self.physics_loss = PhysicsLoss(in_channels_solver=1024)
                    self.physics_enabled = True
                else:
                    self.physics_enabled = False
                
                # Loss weights
                weights = self.config.get('loss_weights', {
                    'ce': 0.5, 'dice': 0.5, 'physics': 0.1, 'smoothness': 0.01
                })
                self.wc = weights.get('ce', 0.5)
                self.wd = weights.get('dice', 0.5)
                self.wp = weights.get('physics', 0.1) if self.physics_enabled else 0.0
                self.ws = weights.get('smoothness', 0.01)
            
            def forward(self, logits, targets, physics_outputs=None, features=None):
                # Basic losses
                loss_ce = self.ce_loss(logits, targets)
                loss_dice = self.dice_loss(logits, targets)
                
                total_loss = self.wc * loss_ce + self.wd * loss_dice
                
                # Physics loss
                if self.physics_enabled and physics_outputs is not None:
                    physics_loss = 0.0
                    for eps, sigma in physics_outputs:
                        # Simplified physics loss calculation
                        physics_constraint = torch.mean(eps**2 + sigma**2)
                        physics_loss += physics_constraint
                    total_loss += self.wp * physics_loss
                
                # Smoothness loss (simplified)
                if features is not None and 'adaptive_smoothing' not in self.ablated_components:
                    smoothness_loss = torch.mean(torch.abs(features[:,:,1:,:] - features[:,:,:-1,:]))
                    smoothness_loss += torch.mean(torch.abs(features[:,:,:,1:] - features[:,:,:,:-1]))
                    total_loss += self.ws * smoothness_loss
                
                return total_loss
        
        return AblatedCombinedLoss(ablated_components, self.config)

    def run_comprehensive_ablation(self, 
                                  dataset_config: Dict[str, Any],
                                  training_config: Dict[str, Any],
                                  num_runs: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive ablation study.
        
        Args:
            dataset_config: Dataset configuration
            training_config: Training configuration
            num_runs: Number of independent runs for each configuration
            
        Returns:
            Comprehensive ablation results
        """
        self.logger.info("Starting comprehensive ablation study")
        
        # Generate all possible component combinations
        component_names = list(self.components.keys())
        all_combinations = []
        
        # Single component ablations
        for component in component_names:
            all_combinations.append([component])
        
        # Pairwise ablations
        for pair in combinations(component_names, 2):
            all_combinations.append(list(pair))
        
        # All components ablated (baseline)
        all_combinations.append(component_names)
        
        # No components ablated (full model)
        all_combinations.append([])
        
        self.logger.info(f"Testing {len(all_combinations)} ablation configurations")
        
        all_results = {}
        
        for config_idx, ablated_components in enumerate(all_combinations):
            config_name = self._generate_config_name(ablated_components)
            self.logger.info(f"Configuration {config_idx + 1}/{len(all_combinations)}: {config_name}")
            
            config_results = []
            
            for run_idx in range(num_runs):
                self.logger.info(f"  Run {run_idx + 1}/{num_runs}")
                
                # Set seed for reproducibility
                set_seed(42 + config_idx * num_runs + run_idx)
                
                # Create ablated model and loss
                model = self.create_ablated_model(ablated_components)
                criterion = self.create_ablated_loss(ablated_components)
                
                # Load data
                train_loader = create_research_dataloader(**dataset_config)
                val_loader = create_research_dataloader(
                    **{**dataset_config, 'shuffle': False, 'augment': False}
                )
                
                # Train and evaluate
                run_result = self._train_and_evaluate_ablated_model(
                    model=model,
                    criterion=criterion,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    training_config=training_config,
                    ablated_components=ablated_components,
                    apply_quantum_noise='quantum_noise' not in ablated_components
                )
                
                run_result.update({
                    'config_name': config_name,
                    'ablated_components': ablated_components,
                    'run': run_idx
                })
                
                config_results.append(run_result)
            
            all_results[config_name] = config_results
        
        # Analyze component contributions
        self.component_contributions = self._analyze_component_contributions(all_results)
        
        # Generate comprehensive report
        self._generate_ablation_report(all_results, self.component_contributions)
        
        self.results = all_results
        return all_results

    def _generate_config_name(self, ablated_components: List[str]) -> str:
        """Generate human-readable configuration name"""
        if not ablated_components:
            return "full_model"
        elif len(ablated_components) == len(self.components):
            return "baseline_unet"
        else:
            ablated_short = [comp.replace('_', '') for comp in ablated_components]
            return f"ablated_{'_'.join(ablated_short)}"

    def _train_and_evaluate_ablated_model(self,
                                         model: nn.Module,
                                         criterion: nn.Module,
                                         train_loader: torch.utils.data.DataLoader,
                                         val_loader: torch.utils.data.DataLoader,
                                         training_config: Dict[str, Any],
                                         ablated_components: List[str],
                                         apply_quantum_noise: bool = True) -> Dict[str, Any]:
        """Train and evaluate ablated model"""
        
        # Setup training
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        # Training metrics
        training_start_time = time.time()
        
        # Training loop
        model.train()
        for epoch in range(training_config.get('epochs', 20)):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Apply quantum noise if enabled
                if apply_quantum_noise:
                    images = quantum_noise_injection(images, T=0.01)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                physics_outputs = None
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    main_output, physics_outputs = outputs[0], outputs[1]
                else:
                    main_output = outputs
                
                # Compute loss
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(main_output, masks)
                else:
                    loss = criterion(main_output, masks, physics_outputs, main_output)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Limit training batches for efficiency
                if batch_idx >= 50:
                    break
        
        training_time = time.time() - training_start_time
        
        # Evaluation
        eval_start_time = time.time()
        metrics = self._evaluate_ablated_model(model, val_loader, apply_quantum_noise)
        eval_time = time.time() - eval_start_time
        
        # Computational analysis
        computational_metrics = self._analyze_computational_overhead(
            model, train_loader, ablated_components
        )
        
        return {
            'metrics': metrics,
            'training_time': training_time,
            'evaluation_time': eval_time,
            'computational': computational_metrics,
            'ablated_components': ablated_components
        }

    def _evaluate_ablated_model(self, 
                               model: nn.Module, 
                               dataloader: torch.utils.data.DataLoader,
                               apply_quantum_noise: bool = False) -> Dict[str, float]:
        """Evaluate ablated model"""
        model.eval()
        
        all_dice_scores = []
        all_iou_scores = []
        all_losses = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(dataloader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Test with and without quantum noise for robustness analysis
                if apply_quantum_noise:
                    images_noisy = quantum_noise_injection(images, T=0.05)
                    outputs = model(images_noisy)
                else:
                    outputs = model(images)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                
                # Compute loss
                loss = criterion(main_output, masks)
                all_losses.append(loss.item())
                
                # Compute metrics
                predictions = torch.argmax(main_output, dim=1)
                
                for i in range(predictions.shape[0]):
                    pred = predictions[i].cpu().numpy()
                    true = masks[i].cpu().numpy()
                    
                    # Dice score
                    dice = self._calculate_dice(pred, true)
                    all_dice_scores.append(dice)
                    
                    # IoU score
                    iou = self._calculate_iou(pred, true)
                    all_iou_scores.append(iou)
                
                # Limit evaluation batches
                if batch_idx >= 20:
                    break
        
        return {
            'dice_mean': float(np.mean(all_dice_scores)),
            'dice_std': float(np.std(all_dice_scores)),
            'iou_mean': float(np.mean(all_iou_scores)),
            'iou_std': float(np.std(all_iou_scores)),
            'loss_mean': float(np.mean(all_losses)),
            'loss_std': float(np.std(all_losses))
        }

    def _calculate_dice(self, pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Dice coefficient"""
        intersection = np.sum(pred * true)
        return (2.0 * intersection) / (np.sum(pred) + np.sum(true) + 1e-8)

    def _calculate_iou(self, pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate IoU score"""
        intersection = np.sum(pred * true)
        union = np.sum(pred) + np.sum(true) - intersection
        return intersection / (union + 1e-8)

    def _analyze_computational_overhead(self, 
                                      model: nn.Module, 
                                      dataloader: torch.utils.data.DataLoader,
                                      ablated_components: List[str]) -> Dict[str, float]:
        """Analyze computational overhead of components"""
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        
        # Memory usage
        sample_input = next(iter(dataloader))[0][:1].to(self.device)
        
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start_time = time.time()
            _ = model(sample_input)
            inference_time = time.time() - start_time
            
            if torch.cuda.is_available():
                memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory_usage = 0.0
        
        # Component-specific overhead estimation
        component_overhead = {}
        for component in self.components:
            if component in ablated_components:
                component_overhead[f"{component}_overhead"] = 0.0
            else:
                # Estimate overhead (simplified)
                if component == 'maxwell_solver':
                    component_overhead[f"{component}_overhead"] = total_params * 0.1
                elif component == 'epure':
                    component_overhead[f"{component}_overhead"] = total_params * 0.05
                else:
                    component_overhead[f"{component}_overhead"] = total_params * 0.02
        
        return {
            'total_parameters': total_params,
            'inference_time_ms': inference_time * 1000,
            'memory_usage_mb': memory_usage,
            **component_overhead
        }

    def _analyze_component_contributions(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze individual component contributions"""
        
        # Get baseline (all components ablated) and full model performance
        baseline_results = results.get('baseline_unet', [])
        full_model_results = results.get('full_model', [])
        
        if not baseline_results or not full_model_results:
            self.logger.warning("Baseline or full model results not found")
            return {}
        
        baseline_dice = np.mean([r['metrics']['dice_mean'] for r in baseline_results])
        full_model_dice = np.mean([r['metrics']['dice_mean'] for r in full_model_results])
        
        component_contributions = {}
        
        # Individual component contributions
        for component in self.components:
            # Find results where only this component is ablated
            ablated_key = f"ablated_{component.replace('_', '')}"
            
            if ablated_key in results:
                ablated_results = results[ablated_key]
                ablated_dice = np.mean([r['metrics']['dice_mean'] for r in ablated_results])
                
                # Contribution = (Full Model - Ablated Model)
                contribution = full_model_dice - ablated_dice
                relative_contribution = contribution / (full_model_dice - baseline_dice + 1e-8)
                
                component_contributions[component] = {
                    'absolute_contribution': float(contribution),
                    'relative_contribution': float(relative_contribution),
                    'full_model_dice': float(full_model_dice),
                    'ablated_dice': float(ablated_dice),
                    'baseline_dice': float(baseline_dice)
                }
        
        # Component interactions (simplified)
        interaction_effects = {}
        component_names = list(self.components.keys())
        
        for i, comp1 in enumerate(component_names):
            for j, comp2 in enumerate(component_names[i+1:], i+1):
                # Find results where both components are ablated
                ablated_both_key = f"ablated_{comp1.replace('_', '')}_{comp2.replace('_', '')}"
                
                if ablated_both_key in results:
                    ablated_both_results = results[ablated_both_key]
                    ablated_both_dice = np.mean([r['metrics']['dice_mean'] for r in ablated_both_results])
                    
                    # Get individual ablation results
                    ablated_comp1_key = f"ablated_{comp1.replace('_', '')}"
                    ablated_comp2_key = f"ablated_{comp2.replace('_', '')}"
                    
                    if ablated_comp1_key in results and ablated_comp2_key in results:
                        ablated_comp1_dice = np.mean([r['metrics']['dice_mean'] for r in results[ablated_comp1_key]])
                        ablated_comp2_dice = np.mean([r['metrics']['dice_mean'] for r in results[ablated_comp2_key]])
                        
                        # Interaction effect = (Individual effects) - (Combined effect)
                        individual_effects = (full_model_dice - ablated_comp1_dice) + (full_model_dice - ablated_comp2_dice)
                        combined_effect = full_model_dice - ablated_both_dice
                        interaction = combined_effect - individual_effects
                        
                        interaction_effects[f"{comp1}_{comp2}"] = float(interaction)
        
        return {
            'component_contributions': component_contributions,
            'interaction_effects': interaction_effects,
            'baseline_performance': float(baseline_dice),
            'full_model_performance': float(full_model_dice),
            'total_improvement': float(full_model_dice - baseline_dice)
        }

    def _generate_ablation_report(self, results: Dict[str, List[Dict]], contributions: Dict[str, Any]):
        """Generate comprehensive ablation study report"""
        
        # Create detailed results DataFrame
        all_rows = []
        for config_name, config_results in results.items():
            for result in config_results:
                row = {
                    'Configuration': config_name,
                    'Run': result['run'],
                    'Dice': result['metrics']['dice_mean'],
                    'IoU': result['metrics']['iou_mean'],
                    'Loss': result['metrics']['loss_mean'],
                    'Training_Time': result['training_time'],
                    'Parameters': result['computational']['total_parameters'],
                    'Memory_MB': result['computational']['memory_usage_mb'],
                    'Ablated_Components': ', '.join(result['ablated_components'])
                }
                all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        df.to_csv(self.output_dir / "detailed_ablation_results.csv", index=False)
        
        # Generate summary statistics
        summary = df.groupby('Configuration').agg({
            'Dice': ['mean', 'std'],
            'IoU': ['mean', 'std'],
            'Loss': ['mean', 'std'],
            'Training_Time': 'mean',
            'Parameters': 'first',
            'Memory_MB': 'mean'
        }).round(4)
        
        summary.to_csv(self.output_dir / "ablation_summary.csv")
        
        # Save component contributions
        with open(self.output_dir / "component_contributions.json", 'w') as f:
            json.dump(contributions, f, indent=2)
        
        # Generate plots
        self._generate_ablation_plots(df, contributions)
        
        self.logger.info(f"Ablation study report generated in {self.output_dir}")

    def _generate_ablation_plots(self, df: pd.DataFrame, contributions: Dict[str, Any]):
        """Generate ablation study plots"""
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Component contribution bar plot
        if 'component_contributions' in contributions:
            plt.figure(figsize=(12, 8))
            comp_contrib = contributions['component_contributions']
            
            components = list(comp_contrib.keys())
            contributions_abs = [comp_contrib[comp]['absolute_contribution'] for comp in components]
            
            plt.bar(components, contributions_abs)
            plt.xlabel('Component')
            plt.ylabel('Absolute Contribution (Dice Score)')
            plt.title('Individual Component Contributions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / "component_contributions.png", dpi=300)
            plt.close()
        
        # 2. Configuration comparison
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, x='Configuration', y='Dice')
        plt.xticks(rotation=45)
        plt.title('Dice Score Comparison Across Ablation Configurations')
        plt.tight_layout()
        plt.savefig(self.output_dir / "configuration_comparison.png", dpi=300)
        plt.close()
        
        # 3. Parameters vs Performance
        plt.figure(figsize=(10, 8))
        summary_df = df.groupby('Configuration').agg({
            'Dice': 'mean',
            'Parameters': 'first'
        }).reset_index()
        
        plt.scatter(summary_df['Parameters'], summary_df['Dice'])
        for i, config in enumerate(summary_df['Configuration']):
            plt.annotate(config, 
                        (summary_df['Parameters'].iloc[i], summary_df['Dice'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Number of Parameters')
        plt.ylabel('Mean Dice Score')
        plt.title('Model Complexity vs Performance')
        plt.tight_layout()
        plt.savefig(self.output_dir / "complexity_vs_performance.png", dpi=300)
        plt.close()
        
        # 4. Component interaction heatmap
        if 'interaction_effects' in contributions and contributions['interaction_effects']:
            interactions = contributions['interaction_effects']
            
            # Create interaction matrix
            components = list(self.components.keys())
            interaction_matrix = np.zeros((len(components), len(components)))
            
            for key, value in interactions.items():
                comp1, comp2 = key.split('_', 1)
                if comp1 in components and comp2 in components:
                    i, j = components.index(comp1), components.index(comp2)
                    interaction_matrix[i, j] = value
                    interaction_matrix[j, i] = value
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(interaction_matrix, 
                       xticklabels=components, 
                       yticklabels=components,
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0)
            plt.title('Component Interaction Effects')
            plt.tight_layout()
            plt.savefig(self.output_dir / "interaction_effects.png", dpi=300)
            plt.close()
        
        self.logger.info("Ablation plots generated successfully")


# Usage example
if __name__ == "__main__":
    config = {
        'n_channels': 1,
        'n_classes': 4,
        'loss_weights': {'ce': 0.5, 'dice': 0.5, 'physics': 0.1, 'smoothness': 0.01}
    }
    
    dataset_config = {
        'dataset_type': 'acdc',
        'data_dir': 'data/raw/ACDC/database/training',
        'batch_size': 4,
        'shuffle': True,
        'augment': True
    }
    
    training_config = {
        'epochs': 5,  # Reduced for testing
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    }
    
    # Run ablation study
    ablation = AblationStudy(config)
    results = ablation.run_comprehensive_ablation(
        dataset_config=dataset_config,
        training_config=training_config,
        num_runs=2  # Reduced for testing
    )
    
    print("Ablation study completed!") 