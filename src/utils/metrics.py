import torch
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(masks, num_classes=4):
    """
    Compute class weights for balanced training
    
    Args:
        masks: List or array of mask data
        num_classes: Number of classes
        
    Returns:
        Array of class weights
    """
    all_labels = []
    for mask in masks:
        all_labels.extend(mask.flatten())
    unique_classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
    print(f"Computed class weights: {class_weights}")
    return class_weights


def evaluate_metrics(model, dataloader, device, num_classes=4):
    """
    Evaluate model performance with comprehensive metrics
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes
        
    Returns:
        Dictionary containing various metrics
    """
    if dataloader is None:
        print("Warning: dataloader is None in evaluate_metrics")
        return {
            'dice_scores': [0.0] * num_classes,
            'iou': [0.0] * num_classes,
            'precision': [0.0] * num_classes,
            'recall': [0.0] * num_classes,
            'f1_score': [0.0] * num_classes
        }

    model.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    dice_s = [0.0] * num_classes
    iou_s = [0.0] * num_classes
    batches = 0

    with torch.no_grad():
        for imgs, tgts in dataloader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            logits, _ = model(imgs)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            batches += 1

            for c in range(num_classes):
                pc_f, tc_f = (preds == c).float().view(-1), (tgts == c).float().view(-1)
                inter = (pc_f * tc_f).sum()
                dice_s[c] += ((2. * inter + 1e-6) / (pc_f.sum() + tc_f.sum() + 1e-6)).item()
                iou_s[c] += ((inter + 1e-6) / (pc_f.sum() + tc_f.sum() - inter + 1e-6)).item()
                tp[c] += inter.item()
                fp[c] += (pc_f.sum() - inter).item()
                fn[c] += (tc_f.sum() - inter).item()

    metrics = {'dice_scores': [], 'iou': [], 'precision': [], 'recall': [], 'f1_score': []}
    if batches > 0:
        for c in range(num_classes):
            metrics['dice_scores'].append(dice_s[c] / batches)
            metrics['iou'].append(iou_s[c] / batches)
            prec, rec = tp[c] / (tp[c] + fp[c] + 1e-6), tp[c] / (tp[c] + fn[c] + 1e-6)
            metrics['precision'].append(prec)
            metrics['recall'].append(rec)
            metrics['f1_score'].append(2 * prec * rec / (prec + rec + 1e-6) if (prec + rec > 0) else 0.0)
    else:
        for _ in range(num_classes):
            [metrics[key].append(0.0) for key in metrics]

    return metrics


def calculate_dice_score(pred, target, num_classes=4, smooth=1e-6):
    """
    Calculate Dice score for segmentation
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        num_classes: Number of classes
        smooth: Smoothing factor
        
    Returns:
        Dice scores per class
    """
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_scores.append(dice.item())
    return dice_scores


def calculate_iou(pred, target, num_classes=4, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) for segmentation
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        num_classes: Number of classes
        smooth: Smoothing factor
        
    Returns:
        IoU scores per class
    """
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
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Dictionary of metrics
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics['dice_scores']))]
    
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Dice Score:  {metrics['dice_scores'][i]:.4f}")
        print(f"  IoU:         {metrics['iou'][i]:.4f}")
        print(f"  Precision:   {metrics['precision'][i]:.4f}")
        print(f"  Recall:      {metrics['recall'][i]:.4f}")
        print(f"  F1 Score:    {metrics['f1_score'][i]:.4f}")
    
    # Calculate averages
    avg_dice = np.mean(metrics['dice_scores'])
    avg_iou = np.mean(metrics['iou'])
    avg_precision = np.mean(metrics['precision'])
    avg_recall = np.mean(metrics['recall'])
    avg_f1 = np.mean(metrics['f1_score'])
    
    print(f"\nAVERAGE METRICS:")
    print(f"  Avg Dice Score:  {avg_dice:.4f}")
    print(f"  Avg IoU:         {avg_iou:.4f}")
    print(f"  Avg Precision:   {avg_precision:.4f}")
    print(f"  Avg Recall:      {avg_recall:.4f}")
    print(f"  Avg F1 Score:    {avg_f1:.4f}")
    print("="*60) 