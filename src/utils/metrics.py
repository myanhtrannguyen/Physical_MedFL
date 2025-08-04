# Federated-Learning/src/utils/metrics.py

from typing import Dict, List, Union
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


def evaluate_metrics(model, dataloader, device, num_classes=4):
    """
    Evaluates model performance and calculates segmentation metrics.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation data
        device (torch.device): Device to run evaluation on
        num_classes (int): Number of classes for segmentation

    Returns:
        dict: Dictionary containing metrics per class:
            - dice_scores: List of Dice scores for each class
            - iou: List of IoU scores for each class
            - precision: List of precision scores for each class
            - recall: List of recall scores for each class
            - f1_score: List of F1 scores for each class
    """

    model.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    dice_s = [0.0] * num_classes
    iou_s = [0.0] * num_classes
    batches = 0

    with torch.no_grad():
        for imgs,tgts in dataloader:
            imgs,tgts = imgs.to(device),tgts.to(device)
            if imgs.size(0) == 0: continue
            
            # Handle model output (could be tuple or tensor)
            model_output = model(imgs)
            if isinstance(model_output, tuple):
                logits = model_output[0]  # Take only logits
            else:
                logits = model_output
                
            preds = torch.argmax(F.softmax(logits,dim=1),dim=1); batches+=1
            for c in range(num_classes):
                pc_f,tc_f=(preds==c).float().view(-1),(tgts==c).float().view(-1); inter=(pc_f*tc_f).sum()
                dice_s[c]+=((2.*inter+1e-6)/(pc_f.sum()+tc_f.sum()+1e-6)).item()
                iou_s[c]+=((inter+1e-6)/(pc_f.sum()+tc_f.sum()-inter+1e-6)).item()
                tp[c]+=inter.item(); fp[c]+=(pc_f.sum()-inter).item(); fn[c]+=(tc_f.sum()-inter).item()
    metrics={'dice_scores':[],'iou':[],'precision':[],'recall':[],'f1_score':[]}
    if batches>0:
        for c in range(num_classes):
            metrics['dice_scores'].append(dice_s[c]/batches); metrics['iou'].append(iou_s[c]/batches)
            prec,rec = tp[c]/(tp[c]+fp[c]+1e-6), tp[c]/(tp[c]+fn[c]+1e-6)
            metrics['precision'].append(prec); metrics['recall'].append(rec)
            metrics['f1_score'].append(2*prec*rec/(prec+rec+1e-6) if (prec+rec > 0) else 0.0)
    else: 
        for _ in range(num_classes): [metrics[key].append(0.0) for key in metrics]
    return metrics