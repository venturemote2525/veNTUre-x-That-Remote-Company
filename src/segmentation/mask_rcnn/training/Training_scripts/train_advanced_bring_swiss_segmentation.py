#!/usr/bin/env python3
"""
Advanced SegFormer Training Script for Bring+Swiss Food Segmentation
Optimized for mask-only segmentation to group similar food regions

This script uses SegFormer-MiT-B3 transformer architecture for semantic segmentation
of the combined Bring+Swiss 7-class food dataset.

Features:
- SegFormer transformer backbone (MiT-B3)
- Combined Focal + Dice Loss for better segmentation
- Advanced data augmentation
- Extensive logging and checkpointing
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = REPO_ROOT / "src" / "segmentation" / "mask_rcnn" / "datasets" / "Modified" / "combined_bring_swiss_7class"
LOG_DIR = REPO_ROOT / "src" / "segmentation" / "mask_rcnn" / "training" / "logs" / "advanced_bring_swiss_segmentation"
CHECKPOINT_DIR = REPO_ROOT / "src" / "segmentation" / "mask_rcnn" / "training" / "checkpoints" / "advanced_bring_swiss_segmentation"

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Advanced SegFormer Configuration for food segmentation
CONFIG = {
    'model_name': 'advanced_bring_swiss_segmentation',
    'backbone': 'nvidia/mit-b3',  # MiT-B3 encoder for good balance of speed/accuracy
    'num_epochs': 10,  # Reduced epochs as requested
    'batch_size': 6,   # Smaller batch size for transformer stability
    'learning_rate': 5e-5,  # Lower LR for pre-trained transformer
    'weight_decay': 0.01,
    'image_size': 512,
    'num_classes': 8,  # 7 food classes + background
    'classes': ['background', 'fruit', 'vegetable', 'carbohydrate', 'protein', 'dairy', 'fat', 'other'],
    'warmup_epochs': 2,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'dice_weight': 0.6,  # Higher weight for Dice loss (better boundaries)
    'focal_weight': 0.4,
    'gradient_clip': 1.0,
    'save_every': 3  # Save checkpoint every N epochs
}

def setup_logging():
    """Setup comprehensive logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"segformer_training_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler  
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('SegFormerTraining')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger, log_file

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in food segmentation."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for better boundary detection in segmentation."""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply softmax to inputs
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for i in range(num_classes):
            input_flat = inputs[:, i].contiguous().view(-1)
            target_flat = targets_one_hot[:, i].contiguous().view(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice = (2 * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
            dice_scores.append(dice)
        
        # Return 1 - average dice (loss)
        return 1 - torch.stack(dice_scores).mean()

class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss for optimal food segmentation."""
    
    def __init__(self, focal_weight=0.4, dice_weight=0.6, focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

class BringSwissDataset(Dataset):
    """Advanced dataset loader for Bring+Swiss food segmentation."""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load COCO annotations
        ann_file = self.root_dir / f"{split}" / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            
        self.coco = COCO(str(ann_file))
        
        # Get image IDs
        self.image_ids = list(self.coco.imgs.keys())
        
        # Load class mapping
        with open(self.root_dir / "class_mapping.json", 'r') as f:
            class_data = json.load(f)
        self.class_names = ['background'] + class_data['target_classes']
        
        print(f"Loaded {len(self.image_ids)} {split} images")
        print(f"Classes: {self.class_names}")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        img_path = self.root_dir / self.split / img_info['file_name']
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create segmentation mask
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                try:
                    # Convert COCO segmentation to mask
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        valid_polygons = [poly for poly in ann['segmentation'] if len(poly) >= 6]
                        if valid_polygons:
                            rles = coco_mask.frPyObjects(valid_polygons, height, width)
                            rle = coco_mask.merge(rles)
                            binary_mask = coco_mask.decode(rle)
                        else:
                            continue
                    else:
                        # RLE format
                        rle = ann['segmentation']
                        binary_mask = coco_mask.decode(rle)
                    
                    # Map category_id to class index
                    cat_id = ann['category_id']
                    if 1 <= cat_id <= 7:  # Valid food class range
                        mask[binary_mask > 0] = cat_id
                except Exception as e:
                    print(f"Skipping annotation due to error: {e}")
                    continue
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()

def get_transforms(split='train', image_size=512):
    """Get advanced data augmentation transforms."""
    
    if split == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.15, 
                rotate_limit=20, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=0, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, 
                contrast_limit=0.25, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=25, 
                val_shift_limit=15, 
                p=0.5
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
            ], p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def calculate_metrics(predictions, targets, num_classes):
    """Calculate comprehensive metrics for segmentation."""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Flatten
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Calculate per-class metrics
    aps = []
    dice_scores = []
    ious = []
    
    for class_id in range(1, num_classes):  # Skip background
        # Binary classification for this class
        pred_binary = (predictions_flat == class_id).astype(int)
        target_binary = (targets_flat == class_id).astype(int)
        
        if target_binary.sum() > 0:  # If class exists in ground truth
            try:
                ap = average_precision_score(target_binary, pred_binary)
                aps.append(ap)
            except:
                aps.append(0.0)
            
            # Dice score
            intersection = (pred_binary * target_binary).sum()
            dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
            dice_scores.append(dice)
            
            # IoU score
            union = pred_binary.sum() + target_binary.sum() - intersection
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        else:
            aps.append(0.0)
            dice_scores.append(0.0)
            ious.append(0.0)
    
    mean_ap = np.mean(aps) if aps else 0.0
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    
    return mean_ap, mean_dice, mean_iou

def train_epoch(model, dataloader, criterion, optimizer, device, logger, epoch):
    """Train for one epoch with detailed logging."""
    model.train()
    total_loss = 0.0
    total_ap = 0.0
    total_dice = 0.0
    total_iou = 0.0
    
    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 15)  # Log 15 times per epoch for more frequent updates
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_values=images)
        logits = outputs.logits
        
        # Resize logits to match mask size if needed
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        
        # Calculate loss
        loss = criterion(logits, masks)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            ap, dice, iou = calculate_metrics(preds, masks, CONFIG['num_classes'])
            total_ap += ap
            total_dice += dice
            total_iou += iou
        
        # Log progress
        if batch_idx % log_interval == 0 or batch_idx == num_batches - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:2d} [{batch_idx:3d}/{num_batches}] "
                f"Loss: {loss.item():.4f} | mAP: {ap:.4f} | Dice: {dice:.4f} | "
                f"IoU: {iou:.4f} | LR: {current_lr:.2e}"
            )
    
    avg_loss = total_loss / num_batches
    avg_ap = total_ap / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_ap, avg_dice, avg_iou

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_ap = 0.0
    total_dice = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images)
            logits = outputs.logits
            
            # Resize logits to match mask size if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # Calculate loss
            loss = criterion(logits, masks)
            total_loss += loss.item()
            
            # Calculate metrics
            preds = torch.argmax(logits, dim=1)
            ap, dice, iou = calculate_metrics(preds, masks, CONFIG['num_classes'])
            total_ap += ap
            total_dice += dice
            total_iou += iou
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_ap = total_ap / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_ap, avg_dice, avg_iou

def main():
    """Main training function."""
    logger, log_file = setup_logging()
    
    logger.info("=" * 80)
    logger.info("ADVANCED BRING+SWISS FOOD SEGMENTATION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {DATASET_ROOT}")
    logger.info(f"Checkpoints: {CHECKPOINT_DIR}")
    logger.info(f"Logs: {log_file}")
    logger.info(f"Configuration: {CONFIG}")
    logger.info("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    train_transform = get_transforms('train', CONFIG['image_size'])
    val_transform = get_transforms('val', CONFIG['image_size'])
    
    try:
        train_dataset = BringSwissDataset(DATASET_ROOT, 'train', train_transform)
        val_dataset = BringSwissDataset(DATASET_ROOT, 'valid', val_transform)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    
    # Create SegFormer model
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            CONFIG['backbone'],
            num_labels=CONFIG['num_classes'],
            id2label={i: CONFIG['classes'][i] for i in range(CONFIG['num_classes'])},
            label2id={CONFIG['classes'][i]: i for i in range(CONFIG['num_classes'])},
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        model.to(device)
        logger.info(f"Created SegFormer model with {CONFIG['backbone']} backbone")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return
    
    # Create loss function and optimizer
    criterion = CombinedLoss(
        focal_weight=CONFIG['focal_weight'],
        dice_weight=CONFIG['dice_weight'],
        focal_alpha=CONFIG['focal_alpha'],
        focal_gamma=CONFIG['focal_gamma']
    )
    
    optimizer = AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG['num_epochs'],
        pct_start=CONFIG['warmup_epochs']/CONFIG['num_epochs']
    )
    
    logger.info("Created optimizer and OneCycleLR scheduler")
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_maps': [],
        'val_maps': [],
        'train_dice': [],
        'val_dice': [],
        'train_ious': [],
        'val_ious': [],
        'class_names': CONFIG['classes'],
        'config': CONFIG
    }
    
    best_val_map = 0.0
    best_val_dice = 0.0
    
    logger.info("Starting training loop...")
    logger.info("-" * 80)
    
    # Training loop
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        logger.info(f"EPOCH {epoch:2d}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss, train_ap, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch
        )
        
        # Validate
        val_loss, val_ap, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_maps'].append(train_ap)
        history['val_maps'].append(val_ap)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_ious'].append(train_iou)
        history['val_ious'].append(val_iou)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch summary
        logger.info("-" * 50)
        logger.info(f"EPOCH {epoch:2d} SUMMARY:")
        logger.info(f"  Train - Loss: {train_loss:.4f}, mAP: {train_ap:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, mAP: {val_ap:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best models
        if val_ap > best_val_map:
            best_val_map = val_ap
            torch.save(model.state_dict(), CHECKPOINT_DIR / "segformer_best_map.pth")
            logger.info(f"  ★ New best mAP: {best_val_map:.4f} - Model saved!")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT_DIR / "segformer_best_dice.pth")
            logger.info(f"  ★ New best Dice: {best_val_dice:.4f} - Model saved!")
        
        # Save checkpoint periodically
        if epoch % CONFIG['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_map': best_val_map,
                'best_val_dice': best_val_dice,
                'history': history,
                'config': CONFIG
            }
            checkpoint_path = CHECKPOINT_DIR / f"segformer_epoch_{epoch:03d}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path.name}")
        
        # Step scheduler
        scheduler.step()
        
        logger.info("-" * 80)
    
    # Save final model and history
    torch.save(model.state_dict(), CHECKPOINT_DIR / "segformer_final.pth")
    
    final_checkpoint = {
        'epoch': CONFIG['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_map': best_val_map,
        'best_val_dice': best_val_dice,
        'history': history,
        'config': CONFIG
    }
    torch.save(final_checkpoint, CHECKPOINT_DIR / "segformer_final_checkpoint.pth")
    
    history_file = CHECKPOINT_DIR / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Best validation mAP: {best_val_map:.4f}")
    logger.info(f"Best validation Dice: {best_val_dice:.4f}")
    logger.info(f"Models saved to: {CHECKPOINT_DIR}")
    logger.info(f"Training history: {history_file}")
    logger.info(f"Training log: {log_file}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()