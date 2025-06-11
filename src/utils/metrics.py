"""
Research-grade metrics for medical image segmentation in federated learning.
Compatible with physics-informed models, ResearchMedicalDataset, and AdaFedAdam.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


def compute_class_weights(masks: Union[List, np.ndarray], num_classes: int = 4) -> np.ndarray:
    """
    Compute class weights for balanced training in medical segmentation.
    
    Args:
        masks: List or array of mask data
        num_classes: Number of classes (default: 4 for ACDC)
        
    Returns:
        Array of class weights for loss balancing
    """
    all_labels = []
    
    # Handle different input formats
    if isinstance(masks, list):
        for mask in masks:
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            all_labels.extend(mask.flatten())
    else:
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        all_labels.extend(masks.flatten())
    
    # Ensure we have all classes represented
    unique_classes = np.arange(num_classes)
    try:
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
        logger.info(f"Computed balanced class weights: {class_weights}")
    except Exception as e:
        logger.warning(f"Failed to compute balanced weights: {e}. Using uniform weights.")
        class_weights = np.ones(num_classes)
    
    return class_weights


def evaluate_model_with_research_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
    return_detailed: bool = False
) -> Dict[str, Any]:
    """
    Research-grade model evaluation with comprehensive metrics.
    Compatible with physics-informed models and ResearchMedicalDataset.
    
    Args:
        model: PyTorch model (supports physics-informed outputs)
        dataloader: DataLoader with ResearchMedicalDataset
        device: Device to run evaluation on
        num_classes: Number of segmentation classes
        class_names: Optional class names for reporting
        return_detailed: Whether to return detailed per-sample metrics
        
    Returns:
        Dictionary containing comprehensive metrics
    """
    if dataloader is None or len(dataloader) == 0:
        logger.warning("Empty or None dataloader in evaluation")
        return _create_empty_metrics(num_classes, class_names)
    
    if class_names is None:
        if num_classes == 4:  # ACDC default
            class_names = ['Background', 'RV', 'Myocardium', 'LV']
        else:
            class_names = [f'Class_{i}' for i in range(num_classes)]
    
    model.eval()
    
    # Initialize metric accumulators
    per_class_metrics = {
        'tp': np.zeros(num_classes),
        'fp': np.zeros(num_classes),
        'fn': np.zeros(num_classes),
    }
    
    total_batches = 0
    total_samples = 0
    sample_details = [] if return_detailed else None
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            try:
                # Move to device
                images = images.to(device).float()
                masks = masks.to(device).long()
                
                if images.size(0) == 0:
                    continue
                
                # Forward pass - handle physics-informed model outputs
                outputs = model(images)
                
                # Handle different model output formats
                if isinstance(outputs, tuple) and len(outputs) >= 1:
                    # Physics-informed model: (logits, bottleneck_features, all_es)
                    logits = outputs[0]
                else:
                    # Standard model output
                    logits = outputs
                
                # Ensure logits is a tensor
                if not torch.is_tensor(logits):
                    logger.error(f"Model output is not a tensor: {type(logits)}")
                    continue
                
                # Get predictions
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Calculate batch metrics
                batch_metrics = _calculate_batch_metrics(
                    predictions, masks, num_classes, class_names
                )
                
                # Accumulate metrics
                for key in per_class_metrics:
                    per_class_metrics[key] += batch_metrics[key]
                
                # Store sample details if requested
                if return_detailed and sample_details is not None:
                    for i in range(images.size(0)):
                        sample_detail = _calculate_sample_metrics(
                            predictions[i], masks[i], num_classes
                        )
                        sample_detail['batch_idx'] = batch_idx
                        sample_detail['sample_idx'] = i
                        sample_details.append(sample_detail)
                
                total_batches += 1
                total_samples += images.size(0)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if total_batches == 0:
        logger.warning("No batches successfully processed")
        return _create_empty_metrics(num_classes, class_names)
    
    # Calculate final metrics
    final_metrics = _finalize_metrics(
        per_class_metrics, total_batches, total_samples, class_names
    )
    
    if return_detailed:
        final_metrics['sample_details'] = sample_details
    
    return final_metrics


def _calculate_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                           num_classes: int, class_names: List[str]) -> Dict[str, np.ndarray]:
    """Calculate metrics for a single batch - OPTIMIZED: Only TP/FP/FN needed."""
    batch_metrics = {
        'tp': np.zeros(num_classes),
        'fp': np.zeros(num_classes),
        'fn': np.zeros(num_classes),
        # REMOVED: dice_sum and iou_sum no longer needed since we calculate from TP/FP/FN
    }
    
    for c in range(num_classes):
        pred_c = (predictions == c).float()
        target_c = (targets == c).float()
        
        # Calculate intersection and union
        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        
        # Only calculate confusion matrix elements (TP/FP/FN)
        # Dice and IoU will be calculated correctly in _finalize_metrics
        batch_metrics['tp'][c] = intersection.item()
        batch_metrics['fp'][c] = (pred_sum - intersection).item()
        batch_metrics['fn'][c] = (target_sum - intersection).item()
    
    return batch_metrics


def _calculate_sample_metrics(prediction: torch.Tensor, target: torch.Tensor, 
                            num_classes: int) -> Dict[str, Any]:
    """Calculate metrics for a single sample."""
    sample_metrics = {
        'dice_scores': [],
        'iou_scores': [],
        'pixel_accuracy': 0.0
    }
    
    # Overall pixel accuracy
    correct_pixels = (prediction == target).sum().item()
    total_pixels = prediction.numel()
    sample_metrics['pixel_accuracy'] = correct_pixels / total_pixels
    
    # Per-class metrics
    for c in range(num_classes):
        pred_c = (prediction == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().item()
        pred_sum = pred_c.sum().item()
        target_sum = target_c.sum().item()
        union = pred_sum + target_sum - intersection
        
        # Dice score
        dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        sample_metrics['dice_scores'].append(dice)
        
        # IoU score
        iou = (intersection + 1e-6) / (union + 1e-6)
        sample_metrics['iou_scores'].append(iou)
    
    return sample_metrics


def _finalize_metrics(per_class_metrics: Dict[str, np.ndarray], total_batches: int,
                     total_samples: int, class_names: List[str]) -> Dict[str, Any]:
    """Finalize accumulated metrics."""
    num_classes = len(class_names)
    
    # CRITICAL FIX: Tính toán Dice và IoU từ tổng TP, FP, FN, không phải từ trung bình các batch
    # Đây là phương pháp macro-averaging chính xác về mặt thống kê
    tp = per_class_metrics['tp']
    fp = per_class_metrics['fp']
    fn = per_class_metrics['fn']

    # FIXED: Tính toán Dice và IoU từ tổng TP, FP, FN đã tích lũy trên toàn bộ dataset
    dice_scores = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou_scores = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    
    # Tính toán Precision và Recall (cách làm ban đầu đã đúng)
    precision_scores = (tp + 1e-6) / (tp + fp + 1e-6)
    recall_scores = (tp + 1e-6) / (tp + fn + 1e-6)
    
    # Tính toán F1-score từ Precision và Recall đã tính
    f1_scores = (2 * precision_scores * recall_scores) / (precision_scores + recall_scores + 1e-6)
    
    # Log detailed metrics for debugging
    logger.debug(f"Accumulated TP: {per_class_metrics['tp']}")
    logger.debug(f"Accumulated FP: {per_class_metrics['fp']}")
    logger.debug(f"Accumulated FN: {per_class_metrics['fn']}")
    logger.debug(f"Final dice scores: {dice_scores}")
    logger.debug(f"Final precision: {precision_scores}")
    logger.debug(f"Final recall: {recall_scores}")
    
    # Calculate aggregated metrics
    metrics = {
        # Per-class metrics - FIXED: All arrays converted to lists for consistency
        'dice_scores': dice_scores.tolist(),
        'iou_scores': iou_scores.tolist(),
        'precision_scores': precision_scores.tolist(),  # Chuyển sang list cho nhất quán
        'recall_scores': recall_scores.tolist(),         # Chuyển sang list cho nhất quán
        'f1_scores': f1_scores.tolist(),                 # Chuyển sang list cho nhất quán
        
        # Aggregated metrics (for FL server reporting)
        'mean_dice': float(np.mean(dice_scores)),
        'mean_iou': float(np.mean(iou_scores)),
        'mean_precision': float(np.mean(precision_scores)),
        'mean_recall': float(np.mean(recall_scores)),
        'mean_f1': float(np.mean(f1_scores)),
        
        # Weighted metrics (excluding background class)
        'mean_dice_fg': float(np.mean(dice_scores[1:])) if num_classes > 1 else float(dice_scores[0]),
        'mean_iou_fg': float(np.mean(iou_scores[1:])) if num_classes > 1 else float(iou_scores[0]),
        
        # Metadata
        'class_names': class_names,
        'num_classes': num_classes,
        'total_batches': total_batches,
        'total_samples': total_samples,
    }
    
    return metrics


def _create_empty_metrics(num_classes: int, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create empty metrics dictionary for error cases."""
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    return {
        'dice_scores': [0.0] * num_classes,
        'iou_scores': [0.0] * num_classes,
        'precision_scores': [0.0] * num_classes,
        'recall_scores': [0.0] * num_classes,
        'f1_scores': [0.0] * num_classes,
        'mean_dice': 0.0,
        'mean_iou': 0.0,
        'mean_precision': 0.0,
        'mean_recall': 0.0,
        'mean_f1': 0.0,
        'mean_dice_fg': 0.0,
        'mean_iou_fg': 0.0,
        'class_names': class_names,
        'num_classes': num_classes,
        'total_batches': 0,
        'total_samples': 0,
    }


def convert_metrics_for_fl_server(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert research metrics to simple scalar format for FL server communication.
    Compatible with AdaFedAdam and Flower serialization requirements.
    
    Args:
        metrics: Comprehensive metrics dictionary
        
    Returns:
        Simplified metrics dictionary with scalar values only
    """
    # FIXED: Improved eval_loss calculation for medical segmentation
    mean_dice = metrics.get('mean_dice', 0.0)
    
    # Use negative log-likelihood style conversion that's more appropriate for FL
    # This gives eval_loss in range [0.1, 2.3] which is more realistic for medical segmentation
    if mean_dice > 0.0:
        eval_loss = -math.log(max(mean_dice, 1e-6))  # Negative log of dice score
    else:
        eval_loss = 2.0  # Reasonable fallback for failed evaluation
    
    # Clamp eval_loss to reasonable range for medical segmentation
    eval_loss = max(0.05, min(eval_loss, 3.0))
    
    fl_metrics = {
        # Primary metrics for FL aggregation - FIXED SCALE
        'eval_loss': float(eval_loss),  # Now properly scaled for medical segmentation
        'mean_dice_score': metrics.get('mean_dice', 0.0),
        'mean_iou_score': metrics.get('mean_iou', 0.0),
        'mean_precision': metrics.get('mean_precision', 0.0),
        'mean_recall': metrics.get('mean_recall', 0.0),
        'mean_f1_score': metrics.get('mean_f1', 0.0),
        
        # Foreground-focused metrics (excluding background)
        'foreground_dice': metrics.get('mean_dice_fg', 0.0),
        'foreground_iou': metrics.get('mean_iou_fg', 0.0),
        
        # Sample information
        'num_examples': float(metrics.get('total_samples', 0)),
        'num_batches': float(metrics.get('total_batches', 0)),
    }
    
    # Add per-class metrics if available
    for i, class_name in enumerate(metrics.get('class_names', [])):
        if i < len(metrics.get('dice_scores', [])):
            fl_metrics[f'dice_{class_name.lower()}'] = float(metrics['dice_scores'][i])
            fl_metrics[f'iou_{class_name.lower()}'] = float(metrics['iou_scores'][i])
    
    return fl_metrics


def print_research_metrics_summary(metrics: Dict[str, Any], title: str = "EVALUATION METRICS") -> None:
    """
    Print a comprehensive, research-grade metrics summary.
    
    Args:
        metrics: Metrics dictionary from evaluate_model_with_research_metrics
        title: Title for the summary
    """
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    
    class_names = metrics.get('class_names', [])
    num_classes = metrics.get('num_classes', 0)
    
    # Per-class metrics
    if class_names and num_classes > 0:
        print("\nPER-CLASS METRICS:")
        print("-" * 80)
        print(f"{'Class':<12} {'Dice':<8} {'IoU':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        print("-" * 80)
        
        for i, class_name in enumerate(class_names):
            if i < len(metrics.get('dice_scores', [])):
                dice = metrics['dice_scores'][i]
                iou = metrics['iou_scores'][i]
                precision = metrics['precision_scores'][i]
                recall = metrics['recall_scores'][i]
                f1 = metrics['f1_scores'][i]
                
                print(f"{class_name:<12} {dice:<8.4f} {iou:<8.4f} {precision:<10.4f} "
                      f"{recall:<8.4f} {f1:<8.4f}")
    
    # Aggregated metrics
    print("\nAGGREGATED METRICS:")
    print("-" * 40)
    print(f"Mean Dice Score:     {metrics.get('mean_dice', 0.0):.4f}")
    print(f"Mean IoU Score:      {metrics.get('mean_iou', 0.0):.4f}")
    print(f"Mean Precision:      {metrics.get('mean_precision', 0.0):.4f}")
    print(f"Mean Recall:         {metrics.get('mean_recall', 0.0):.4f}")
    print(f"Mean F1 Score:       {metrics.get('mean_f1', 0.0):.4f}")
    
    # Foreground metrics (clinical focus)
    print(f"\nFOREGROUND FOCUS (Clinical Relevance):")
    print("-" * 40)
    print(f"Foreground Dice:     {metrics.get('mean_dice_fg', 0.0):.4f}")
    print(f"Foreground IoU:      {metrics.get('mean_iou_fg', 0.0):.4f}")
    
    # Sample information
    print(f"\nDATASET INFORMATION:")
    print("-" * 40)
    print(f"Total Samples:       {metrics.get('total_samples', 0)}")
    print(f"Total Batches:       {metrics.get('total_batches', 0)}")
    print(f"Classes:             {num_classes}")
    
    print("="*80)


# Legacy compatibility functions (for backward compatibility)
def evaluate_metrics(model, dataloader, device, num_classes=4):
    """Legacy function - use evaluate_model_with_research_metrics instead."""
    logger.warning("evaluate_metrics is deprecated. Use evaluate_model_with_research_metrics instead.")
    
    result = evaluate_model_with_research_metrics(model, dataloader, device, num_classes)
    
    # Convert to legacy format
    return {
        'dice_scores': result.get('dice_scores', []),
        'iou': result.get('iou_scores', []),
        'precision': result.get('precision_scores', []),
        'recall': result.get('recall_scores', []),
        'f1_score': result.get('f1_scores', [])
    }


def calculate_dice_score(pred, target, num_classes=4, smooth=1e-6):
    """Legacy function - maintained for compatibility."""
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_scores.append(dice.item())
    return dice_scores


def calculate_iou(pred, target, num_classes=4, smooth=1e-6):
    """Legacy function - maintained for compatibility."""
    iou_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    return iou_scores


def print_metrics_summary(metrics, class_names=None):
    """Legacy function - use print_research_metrics_summary instead."""
    logger.warning("print_metrics_summary is deprecated. Use print_research_metrics_summary instead.")
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics['dice_scores']))]
    
    # Convert to new format and print
    new_metrics = {
        'dice_scores': metrics.get('dice_scores', []),
        'iou_scores': metrics.get('iou', []),
        'precision_scores': metrics.get('precision', []),
        'recall_scores': metrics.get('recall', []),
        'f1_scores': metrics.get('f1_score', []),
        'class_names': class_names,
        'num_classes': len(class_names),
        'mean_dice': np.mean(metrics.get('dice_scores', [])),
        'mean_iou': np.mean(metrics.get('iou', [])),
        'mean_precision': np.mean(metrics.get('precision', [])),
        'mean_recall': np.mean(metrics.get('recall', [])),
        'mean_f1': np.mean(metrics.get('f1_score', [])),
    }
    
    print_research_metrics_summary(new_metrics, "LEGACY METRICS SUMMARY")


# Export new research-grade functions
__all__ = [
    'compute_class_weights',
    'evaluate_model_with_research_metrics',
    'convert_metrics_for_fl_server',
    'print_research_metrics_summary',
    # Legacy functions
    'evaluate_metrics',
    'calculate_dice_score',
    'calculate_iou',
    'print_metrics_summary'
] 