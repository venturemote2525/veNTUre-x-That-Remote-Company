#!/usr/bin/env python3
"""
Visualization utilities for Mask R-CNN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import cv2
from PIL import Image
from pathlib import Path

def save_prediction_images(model, dataset, device, save_dir, num_samples=5, class_names=None):
    """Save prediction visualization images"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"Saving {num_samples} prediction visualizations to {save_dir}")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            image, target = dataset[i]
            
            # Get prediction
            prediction = model([image.to(device)])[0]
            
            # Create visualization
            fig = visualize_prediction(
                image, target, prediction, 
                class_names=class_names,
                confidence_threshold=0.5
            )
            
            # Save
            save_path = save_dir / f"prediction_{i:03d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print("Prediction visualizations saved!")

def visualize_prediction(image, target, prediction, class_names=None, confidence_threshold=0.5):
    """Visualize model prediction vs ground truth"""
    
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = np.array(image)
    
    # Ensure image values are in [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image_np)
    axes[1].set_title('Ground Truth')
    
    if target['boxes'].numel() > 0:
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        
        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            x1, y1, x2, y2 = box
            
            # Bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            axes[1].add_patch(rect)
            
            # Label
            label_name = get_class_name(label, class_names)
            axes[1].text(x1, y1-5, label_name, color='green', 
                        fontsize=10, weight='bold')
            
            # Mask if available
            if 'masks' in target and target['masks'].numel() > 0:
                mask = target['masks'][i].cpu().numpy()
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    contour = contour.squeeze(1)
                    if len(contour) > 2:
                        polygon = Polygon(contour, closed=True, fill=False,
                                        edgecolor='lime', linewidth=2, alpha=0.8)
                        axes[1].add_patch(polygon)
    
    axes[1].axis('off')
    
    # Predictions
    axes[2].imshow(image_np)
    axes[2].set_title(f'Predictions (conf > {confidence_threshold})')
    
    if prediction['boxes'].numel() > 0:
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        
        # Filter by confidence
        keep_indices = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[keep_indices]
        pred_scores = pred_scores[keep_indices]
        pred_labels = pred_labels[keep_indices]
        
        for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            x1, y1, x2, y2 = box
            
            # Bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[2].add_patch(rect)
            
            # Label with confidence
            label_name = get_class_name(label, class_names)
            axes[2].text(x1, y1-5, f'{label_name}: {score:.2f}', 
                        color='red', fontsize=10, weight='bold')
            
            # Mask if available
            if 'masks' in prediction and prediction['masks'].numel() > 0:
                if i < len(prediction['masks']):
                    mask = prediction['masks'][i].cpu().numpy()
                    # Handle different mask formats
                    if len(mask.shape) == 3:
                        mask = mask[0]  # Take first channel if 3D
                    
                    # Threshold mask
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours:
                        contour = contour.squeeze(1)
                        if len(contour) > 2:
                            polygon = Polygon(contour, closed=True, fill=False,
                                            edgecolor='yellow', linewidth=2, alpha=0.8)
                            axes[2].add_patch(polygon)
    
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def get_class_name(label, class_names):
    """Get class name from label index"""
    if class_names and isinstance(class_names, list):
        if 0 <= label < len(class_names):
            return class_names[label]
    return f"class_{label}"

def visualize_training_history(train_losses, val_losses, val_maps, save_path=None):
    """Visualize training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training loss
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    if len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        axes[0].plot(val_epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Validation loss
    if len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        axes[1].plot(val_epochs, val_losses, 'r-')
        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'No validation data', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    # Validation mAP
    if len(val_maps) > 0:
        val_epochs = range(1, len(val_maps) + 1)
        axes[2].plot(val_epochs, val_maps, 'g-')
        axes[2].set_title('Validation mAP')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mAP')
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, 'No mAP data', 
                    ha='center', va='center', transform=axes[2].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    return fig

def create_detection_summary(predictions, targets, class_names=None):
    """Create a summary of detection results"""
    
    total_predictions = len(predictions)
    total_targets = len(targets)
    
    # Count objects per class
    pred_counts = {}
    target_counts = {}
    
    for pred in predictions:
        for label in pred['labels']:
            class_name = get_class_name(label.item(), class_names)
            pred_counts[class_name] = pred_counts.get(class_name, 0) + 1
    
    for target in targets:
        for label in target['labels']:
            class_name = get_class_name(label.item(), class_names)
            target_counts[class_name] = target_counts.get(class_name, 0) + 1
    
    print("\nDETECTION SUMMARY")
    print("="*40)
    print(f"Total images: {total_predictions}")
    print(f"Total predicted objects: {sum(pred_counts.values())}")
    print(f"Total ground truth objects: {sum(target_counts.values())}")
    
    print("\nPer-class counts:")
    all_classes = set(list(pred_counts.keys()) + list(target_counts.keys()))
    
    for class_name in sorted(all_classes):
        pred_count = pred_counts.get(class_name, 0)
        target_count = target_counts.get(class_name, 0)
        print(f"  {class_name}: {pred_count} predicted, {target_count} ground truth")
    
    print("="*40)

def plot_loss_components(loss_history, save_path=None):
    """Plot individual loss components if available"""
    
    if not isinstance(loss_history, dict):
        print("Loss history should be a dict with component losses")
        return
    
    components = list(loss_history.keys())
    num_components = len(components)
    
    if num_components == 0:
        return
    
    # Create subplot grid
    cols = min(3, num_components)
    rows = (num_components + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (component, values) in enumerate(loss_history.items()):
        if i < len(axes):
            epochs = range(1, len(values) + 1)
            axes[i].plot(epochs, values)
            axes[i].set_title(f'{component.replace("_", " ").title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True)
    
    # Hide unused subplots
    for i in range(num_components, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss components plot saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Create dummy data for testing
    train_losses = [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6]
    val_losses = [2.8, 2.2, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9]
    val_maps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72]
    
    fig = visualize_training_history(train_losses, val_losses, val_maps)
    plt.show()
    
    print("Visualization test completed!")