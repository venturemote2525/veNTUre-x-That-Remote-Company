#!/usr/bin/env python3
"""
Evaluation metrics for Mask R-CNN
"""

import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools import mask as coco_mask
from pycocotools.cocoeval import COCOeval
import json

def calculate_mAP(predictions, targets, num_classes, iou_threshold=0.5):
    """Calculate mean Average Precision (mAP) for object detection"""
    try:
        # Use torchmetrics for mAP calculation
        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox"  # Can be "bbox", "segm", or "keypoints" 
        )
        
        # Format predictions and targets for torchmetrics
        preds = []
        targets_formatted = []
        
        for pred, target in zip(predictions, targets):
            # Predictions
            pred_dict = {
                'boxes': pred['boxes'],
                'scores': pred['scores'],
                'labels': pred['labels']
            }
            preds.append(pred_dict)
            
            # Targets
            target_dict = {
                'boxes': target['boxes'], 
                'labels': target['labels']
            }
            targets_formatted.append(target_dict)
        
        # Update metric
        metric.update(preds, targets_formatted)
        
        # Compute mAP
        result = metric.compute()
        map_value = result['map'].item()
        
        return map_value
        
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        return 0.0

def calculate_mask_mAP(predictions, targets, num_classes):
    """Calculate mAP for instance segmentation masks"""
    try:
        # Use torchmetrics for mask mAP
        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="segm"
        )
        
        preds = []
        targets_formatted = []
        
        for pred, target in zip(predictions, targets):
            # Predictions
            pred_dict = {
                'boxes': pred['boxes'],
                'scores': pred['scores'], 
                'labels': pred['labels'],
                'masks': pred['masks']
            }
            preds.append(pred_dict)
            
            # Targets
            target_dict = {
                'boxes': target['boxes'],
                'labels': target['labels'],
                'masks': target['masks']
            }
            targets_formatted.append(target_dict)
        
        metric.update(preds, targets_formatted)
        result = metric.compute()
        
        return result['map'].item()
        
    except Exception as e:
        print(f"Error calculating mask mAP: {e}")
        return 0.0

def calculate_iou(mask1, mask2):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def evaluate_model(model, data_loader, device, num_classes):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Move to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = model(images)
            
            # Calculate loss (temporarily switch to training mode)
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            model.eval()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            num_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(num_batches, 1)
    bbox_map = calculate_mAP(all_predictions, all_targets, num_classes)
    
    # Try to calculate mask mAP if masks are available
    mask_map = 0.0
    try:
        if len(all_predictions) > 0 and 'masks' in all_predictions[0]:
            mask_map = calculate_mask_mAP(all_predictions, all_targets, num_classes)
    except:
        pass
    
    results = {
        'loss': avg_loss,
        'bbox_mAP': bbox_map,
        'mask_mAP': mask_map,
        'num_samples': len(all_predictions)
    }
    
    return results

def compute_ap_per_class(predictions, targets, num_classes):
    """Compute Average Precision per class"""
    
    class_aps = {}
    
    for class_id in range(1, num_classes):  # Skip background (0)
        class_preds = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # Filter predictions for this class
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                class_pred = {
                    'boxes': pred['boxes'][class_mask],
                    'scores': pred['scores'][class_mask],
                    'labels': pred['labels'][class_mask]
                }
                class_preds.append(class_pred)
            else:
                class_preds.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long)
                })
            
            # Filter targets for this class
            target_mask = target['labels'] == class_id
            if target_mask.any():
                class_target = {
                    'boxes': target['boxes'][target_mask],
                    'labels': target['labels'][target_mask]
                }
                class_targets.append(class_target)
            else:
                class_targets.append({
                    'boxes': torch.empty((0, 4)),
                    'labels': torch.empty(0, dtype=torch.long)
                })
        
        # Calculate AP for this class
        if len(class_preds) > 0:
            class_ap = calculate_mAP(class_preds, class_targets, 2)  # Binary classification
            class_aps[f'class_{class_id}'] = class_ap
    
    return class_aps

def print_evaluation_results(results, class_names=None):
    """Pretty print evaluation results"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Number of samples: {results['num_samples']}")
    print(f"Average Loss: {results['loss']:.4f}")
    print(f"Bounding Box mAP: {results['bbox_mAP']:.4f}")
    
    if results['mask_mAP'] > 0:
        print(f"Mask mAP: {results['mask_mAP']:.4f}")
    
    # Print class-wise results if available
    if 'class_aps' in results:
        print(f"\nPer-Class Average Precision:")
        for class_name, ap in results['class_aps'].items():
            if class_names and isinstance(class_names, list):
                try:
                    class_idx = int(class_name.split('_')[-1])
                    if class_idx < len(class_names):
                        display_name = class_names[class_idx]
                    else:
                        display_name = class_name
                except:
                    display_name = class_name
            else:
                display_name = class_name
            print(f"  {display_name}: {ap:.4f}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test the metrics with dummy data
    print("Testing metrics...")
    
    # Dummy predictions and targets
    predictions = [
        {
            'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            'scores': torch.tensor([0.9, 0.8]),
            'labels': torch.tensor([1, 2])
        }
    ]
    
    targets = [
        {
            'boxes': torch.tensor([[12, 12, 48, 48], [65, 65, 95, 95]]),
            'labels': torch.tensor([1, 2])
        }
    ]
    
    map_value = calculate_mAP(predictions, targets, 3)
    print(f"Test mAP: {map_value:.4f}")