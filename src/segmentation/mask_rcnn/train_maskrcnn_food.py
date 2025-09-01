#!/usr/bin/env python3
"""
PyTorch Mask R-CNN Training Script for Food Segmentation
Based on the shapes training notebook but adapted for food data with ResNet-50
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models import resnet18, resnet50
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import json
import time
from pathlib import Path
import argparse
from datetime import datetime
import logging

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.food_dataset import FoodSegmentationDataset
from utils.config import FoodMaskRCNNConfig, get_config
from utils.metrics import calculate_mAP, evaluate_model
from utils.visualization import save_prediction_images

class FoodMaskRCNNTrainer:
    """PyTorch Mask R-CNN trainer for food segmentation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(f"Initializing trainer on {self.device}")
        self.logger.info(f"Config: {config.name}")
        print(f"Initializing trainer on {self.device}")
        print(f"Config: {config.name}")
    
    def setup_logging(self):
        """Setup logging for training monitoring"""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('MaskRCNN_Training')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for detailed logs
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Training logger initialized")
        print(f"Training logs will be saved to: {log_file}")
    
    def build_model(self):
        """Build Mask R-CNN with configurable ResNet backbone"""
        print(f"Building Mask R-CNN with {self.config.backbone.upper()} backbone...")
        
        if self.config.backbone == 'resnet50':
            # Use built-in ResNet-50 Mask R-CNN
            self.model = maskrcnn_resnet50_fpn(
                pretrained=True,
                progress=True,
                pretrained_backbone=True
            )
        elif self.config.backbone in ['resnet18', 'resnet101', 'resnet152']:
            # Build custom Mask R-CNN with specified ResNet backbone
            backbone = resnet_fpn_backbone(self.config.backbone, pretrained=True)
            self.model = MaskRCNN(
                backbone,
                num_classes=self.config.num_classes,
                min_size=self.config.image_min_size,
                max_size=self.config.image_max_size
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone}")
            
        # For ResNet-50, we need to adjust the heads for custom classes
        if self.config.backbone == 'resnet50':
            # Replace the classifier head
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, 
                self.config.num_classes
            )
            
            # Replace the mask predictor head
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                self.config.num_classes
            )
        
        self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders"""
        print("Setting up data loaders...")
        
        # Training dataset
        train_dataset = FoodSegmentationDataset(
            root_dir=self.config.data_root,
            split='train',
            transforms=self.config.train_transforms,
            class_names=self.config.class_names
        )
        
        # Validation dataset  
        val_dataset = FoodSegmentationDataset(
            root_dir=self.config.data_root,
            split='val',
            transforms=self.config.val_transforms,
            class_names=self.config.class_names
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Data loaders - Windows multiprocessing fix
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Force single-threaded on Windows
            collate_fn=self.collate_fn,
            pin_memory=False,  # Disable for memory safety on RTX 2060
            persistent_workers=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Keep at 1 for memory safety
            shuffle=False,
            num_workers=0,  # Force single-threaded on Windows
            collate_fn=self.collate_fn,
            pin_memory=False,  # Disable to save memory
            persistent_workers=False
        )
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for Mask R-CNN"""
        return tuple(zip(*batch))
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:  # Adam
            self.optimizer = optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Learning rate scheduler
        if self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        
        print(f"Optimizer: {self.config.optimizer}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Scheduler: {self.config.scheduler}")
    
    def train_epoch(self, epoch):
        """Train for one epoch with aggressive memory optimization"""
        self.model.train()
        
        total_loss = 0
        num_batches = len(self.train_loader)
        accumulation_steps = 8  # Higher accumulation for memory efficiency
        processed_batches = 0
        
        self.optimizer.zero_grad()
        
        # Initialize gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        # Clear cache before starting
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            try:
                # Clear cache every few batches
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                
                # Resize images to smaller resolution to save memory
                resized_images = []
                for img in images:
                    # Resize to max 800px on longest side
                    h, w = img.shape[-2:]
                    max_size = 600  # Reduced from 800 for RTX 2060
                    if max(h, w) > max_size:
                        scale = max_size / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        img = torch.nn.functional.interpolate(
                            img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
                        ).squeeze(0)
                    resized_images.append(img.to(self.device))
                
                # Resize targets accordingly
                resized_targets = []
                for i, target in enumerate(targets):
                    orig_h, orig_w = images[i].shape[-2:]
                    new_h, new_w = resized_images[i].shape[-2:]
                    scale_x = new_w / orig_w
                    scale_y = new_h / orig_h
                    
                    new_target = {}
                    for k, v in target.items():
                        if k == 'boxes':
                            # Scale bounding boxes
                            boxes = v.clone()
                            boxes[:, [0, 2]] *= scale_x
                            boxes[:, [1, 3]] *= scale_y
                            new_target[k] = boxes.to(self.device)
                        elif k == 'masks':
                            # Resize masks
                            masks = torch.nn.functional.interpolate(
                                v.float().unsqueeze(1), size=(new_h, new_w), mode='nearest'
                            ).squeeze(1).bool()
                            new_target[k] = masks.to(self.device)
                        else:
                            new_target[k] = v.to(self.device)
                    resized_targets.append(new_target)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(resized_images, resized_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Scale loss by accumulation steps
                    losses = losses / accumulation_steps
                
                # Check for NaN/inf losses
                if not torch.isfinite(losses):
                    print(f"Warning: Non-finite loss detected: {losses.item()}")
                    torch.cuda.empty_cache()
                    continue
                
                # Backward pass with gradient scaling
                scaler.scale(losses).backward()
                
                processed_batches += 1
                
                # Perform optimizer step every accumulation_steps
                if processed_batches % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # Gradient clipping
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Lower max_norm
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Clear cache after optimizer step
                    torch.cuda.empty_cache()
                
                # Store unscaled loss for logging
                actual_loss = losses.item() * accumulation_steps
                total_loss += actual_loss
                
                # Log progress more frequently for 70 epochs
                if (batch_idx + 1) % max(1, self.config.print_freq // 2) == 0:
                    avg_loss = total_loss / processed_batches
                    progress = (batch_idx + 1) / num_batches * 100
                    print(f"Epoch [{epoch+1}/{self.config.epochs}], "
                          f"Batch [{batch_idx+1}/{num_batches}] ({progress:.1f}%), "
                          f"Loss: {actual_loss:.4f}, "
                          f"Avg Loss: {avg_loss:.4f}")
                    
                    # Log to file as well
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, "
                                   f"Loss: {actual_loss:.4f}, Avg Loss: {avg_loss:.4f}")
                
                # Free memory
                del resized_images, resized_targets, loss_dict, losses
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM Error at batch {batch_idx+1}. Clearing cache and skipping.")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()  # Clear gradients
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"Training error at batch {batch_idx+1}: {e}")
                torch.cuda.empty_cache()
                continue
        
        if processed_batches == 0:
            print("Warning: No batches were successfully processed!")
            return 0.0
        
        avg_loss = total_loss / processed_batches
        self.train_losses.append(avg_loss)
        
        # Final cache clear
        torch.cuda.empty_cache()
        
        # Log epoch completion with more details
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{self.config.epochs} COMPLETED")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"Processed Batches: {processed_batches}/{num_batches}")
        print(f"Success Rate: {processed_batches/num_batches*100:.1f}%")
        print(f"{'='*50}")
        
        # Enhanced file logging
        self.logger.info(f"Epoch {epoch+1}/{self.config.epochs} completed")
        self.logger.info(f"Average Training Loss: {avg_loss:.4f}")
        self.logger.info(f"Processed Batches: {processed_batches}/{num_batches}")
        self.logger.info(f"Success Rate: {processed_batches/num_batches*100:.1f}%")
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch with memory-efficient mAP calculation"""
        self.model.eval()
        
        total_loss = 0
        processed_batches = 0
        predictions = []
        ground_truths = []
        
        # Limit validation samples for memory efficiency
        max_val_samples = 15  # Reduced further for memory
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                if batch_idx >= max_val_samples:
                    break
                    
                try:
                    # Clear memory before processing
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Move to device
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Calculate loss
                    self.model.train()
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += losses.item()
                    processed_batches += 1
                    
                    # Get predictions for mAP calculation
                    self.model.eval()
                    outputs = self.model(images)
                    
                    # Store predictions and targets for mAP calculation
                    for i, (output, target) in enumerate(zip(outputs, targets)):
                        # Keep all predictions but convert to tensors for torchmetrics
                        if len(output['scores']) > 0:
                            pred = {
                                'boxes': output['boxes'].cpu(),
                                'scores': output['scores'].cpu(),
                                'labels': output['labels'].cpu()
                            }
                            # Debug: print prediction stats
                            if batch_idx == 0:  # Only first batch to avoid spam
                                max_score = output['scores'].max().item() if len(output['scores']) > 0 else 0
                                num_preds = len(output['scores'])
                                print(f"Debug: Max score: {max_score:.4f}, Total predictions: {num_preds}")
                        else:
                            pred = {'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.int64)}
                            
                        gt = {
                            'boxes': target['boxes'].cpu(),
                            'labels': target['labels'].cpu()
                        }
                        
                        predictions.append(pred)
                        ground_truths.append(gt)
                    
                    # Clear tensors immediately
                    del images, targets, loss_dict, losses, outputs
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Validation OOM at batch {batch_idx}, clearing cache and skipping...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        print(f"Validation error: {e}")
                        break
                except Exception as e:
                    print(f"Validation error at batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(1, processed_batches)
        
        # Calculate simplified mAP if we have predictions
        val_map = 0.0
        if len(predictions) > 0 and len(ground_truths) > 0:
            try:
                val_map = self.calculate_simple_map(predictions, ground_truths)
            except Exception as e:
                print(f"mAP calculation failed: {e}")
                val_map = 0.0
        
        self.val_losses.append(avg_loss)
        self.val_maps.append(val_map)
        
        print(f"Validation - Loss: {avg_loss:.4f}, mAP: {val_map:.4f}")
        self.logger.info(f"Epoch {epoch+1} Validation - Loss: {avg_loss:.4f}, mAP: {val_map:.4f}")
        
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        return avg_loss, val_map
    
    def calculate_simple_map(self, predictions, ground_truths, iou_threshold=0.5):
        """Calculate simplified mAP using IoU matching"""
        import numpy as np
        
        def bbox_iou(box1, box2):
            """Calculate IoU between two bounding boxes"""
            # Convert to format: [x1, y1, x2, y2]
            if len(box1) == 0 or len(box2) == 0:
                return 0.0
                
            # Calculate intersection
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Calculate AP for each class
        class_aps = []
        unique_classes = set()
        
        # Collect all unique classes
        for gt in ground_truths:
            if len(gt['labels']) > 0:
                unique_classes.update(gt['labels'])
        
        if len(unique_classes) == 0:
            return 0.0
            
        for class_id in unique_classes:
            # Collect all predictions and ground truths for this class
            class_preds = []
            class_gts = []
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Get predictions for this class
                if len(pred['labels']) > 0:
                    class_mask = pred['labels'] == class_id
                    if np.any(class_mask):
                        class_pred_boxes = pred['boxes'][class_mask]
                        class_pred_scores = pred['scores'][class_mask]
                        for j, (box, score) in enumerate(zip(class_pred_boxes, class_pred_scores)):
                            class_preds.append((score, box, i))  # (score, box, image_idx)
                
                # Get ground truths for this class
                if len(gt['labels']) > 0:
                    gt_mask = gt['labels'] == class_id
                    if np.any(gt_mask):
                        class_gt_boxes = gt['boxes'][gt_mask]
                        for box in class_gt_boxes:
                            class_gts.append((box, i))  # (box, image_idx)
            
            if len(class_preds) == 0 or len(class_gts) == 0:
                continue
                
            # Sort predictions by confidence
            class_preds.sort(key=lambda x: x[0], reverse=True)
            
            # Calculate precision-recall
            tp = 0
            fp = 0
            matched_gts = set()
            
            for score, pred_box, img_idx in class_preds:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching GT in same image
                for gt_idx, (gt_box, gt_img_idx) in enumerate(class_gts):
                    if gt_img_idx == img_idx and (gt_img_idx, gt_idx) not in matched_gts:
                        iou = bbox_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gts.add((img_idx, best_gt_idx))
                else:
                    fp += 1
            
            # Calculate AP (simplified version)
            if tp + fp > 0:
                precision = tp / (tp + fp)
                recall = tp / len(class_gts) if len(class_gts) > 0 else 0
                ap = precision * recall  # Simplified AP calculation
                class_aps.append(ap)
        
        # Return mean AP
        return np.mean(class_aps) if len(class_aps) > 0 else 0.0
    
    def save_checkpoint(self, epoch, val_map, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(getattr(self.config, 'model_dir', checkpoint_dir))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_map': val_map,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maps': self.val_maps,
            'config': self.config.__dict__
        }
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        # Also export latest model weights under models/<name>
        latest_weights = model_dir / f"{self.config.name}_latest_{self.config.backbone}.pth"
        torch.save(self.model.state_dict(), latest_weights)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            # Mirror best weights to models/<name>
            best_weights = model_dir / f"{self.config.name}_best_{self.config.backbone}.pth"
            torch.save(self.model.state_dict(), best_weights)
            print(f"New best model saved with mAP: {val_map:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.save_freq == 0:
            epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, epoch_path)
    
    def save_training_history(self):
        """Save training history with JSON-serializable config"""
        # Make config JSON-friendly by stringifying non-serializable values
        cfg = {}
        for k, v in self.config.__dict__.items():
            if k in ('train_transforms', 'val_transforms'):
                cfg[k] = str(v)
                continue
            try:
                import json as _json
                _json.dumps(v)
                cfg[k] = v
            except Exception:
                cfg[k] = str(v)

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maps': self.val_maps,
            'config': cfg,
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure directory exists even if removed mid-run
        chk_dir = Path(self.config.checkpoint_dir)
        chk_dir.mkdir(parents=True, exist_ok=True)
        history_path = chk_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def freeze_backbone(self):
        """Freeze backbone layers for heads-only training"""
        print("Freezing backbone layers for heads-only training...")
        
        # Freeze backbone
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = False
            
        # Freeze FPN if present
        if hasattr(self.model, 'fpn'):
            for name, param in self.model.fpn.named_parameters():
                param.requires_grad = False
        
        # Keep head layers trainable
        for name, param in self.model.roi_heads.named_parameters():
            param.requires_grad = True
            
        for name, param in self.model.rpn.named_parameters():
            param.requires_grad = True
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters (heads only): {trainable_params:,}")
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for full fine-tuning"""
        print("Unfreezing all layers for full fine-tuning...")
        
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters (all layers): {trainable_params:,}")
    
    def train(self):
        """Two-stage training: heads-only then full fine-tuning with crash protection"""
        print(f"\n{'='*50}")
        print("Starting Food Mask R-CNN Two-Stage Training")
        print(f"{'='*50}")
        
        best_val_map = 0.0
        start_time = time.time()
        
        # Stage 1: Train heads only
        print(f"\n{'='*20} STAGE 1: HEADS ONLY {'='*20}")
        self.freeze_backbone()
        
        # Adjust learning rate for heads training
        heads_lr = self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = heads_lr
        
        heads_epochs = self.config.heads_epochs if hasattr(self.config, 'heads_epochs') else max(1, self.config.epochs // 4)
        print(f"Training heads for {heads_epochs} epochs with LR: {heads_lr}")
        
        for epoch in range(heads_epochs):
            print(f"\nStage 1 - Epoch {epoch+1}/{heads_epochs}")
            print("-" * 30)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_map = self.validate_epoch(epoch)
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_map > best_val_map
            if is_best:
                best_val_map = val_map
            
            self.save_checkpoint(epoch, val_map, is_best)
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"Stage 1 Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val mAP: {val_map:.4f}")
            print(f"  Best mAP: {best_val_map:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Elapsed Time: {elapsed/3600:.2f}h")
        
        # Stage 2: Fine-tune all layers with reduced batch size for large models
        print(f"\n{'='*20} STAGE 2: ALL LAYERS {'='*20}")
        self.unfreeze_all_layers()
        
        # Auto-reduce batch size for large models to prevent OOM
        if hasattr(self.config, 'stage2_batch_size') and self.config.stage2_batch_size:
            stage2_batch_size = self.config.stage2_batch_size
        else:
            # Auto-calculate reduced batch size based on backbone
            if 'resnet152' in self.config.backbone:
                stage2_batch_size = max(1, self.config.batch_size // 6)  # Aggressive reduction
            elif 'resnet101' in self.config.backbone:
                stage2_batch_size = max(1, self.config.batch_size // 4)  # Moderate reduction
            else:
                stage2_batch_size = max(1, self.config.batch_size // 2)  # Conservative reduction
        
        print(f"Reducing batch size from {self.config.batch_size} to {stage2_batch_size} for Stage 2")
        
        # Recreate data loader with reduced batch size
        if stage2_batch_size != self.config.batch_size:
            from torch.utils.data import DataLoader
            from utils.food_dataset import FoodSegmentationDataset
            
            train_dataset = FoodSegmentationDataset(
                root_dir=self.config.data_root,
                split='train',
                transforms=self.config.train_transforms,
                class_names=self.config.class_names
            )
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=stage2_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=self.collate_fn,
                pin_memory=False,
                persistent_workers=False
            )
        
        # Lower learning rate for full fine-tuning
        finetune_lr = self.config.learning_rate / 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = finetune_lr
        
        # Reset scheduler for stage 2
        remaining_epochs = self.config.epochs - heads_epochs
        if self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_epochs
            )
        
        print(f"Fine-tuning all layers for {remaining_epochs} epochs with LR: {finetune_lr}")
        
        for epoch in range(heads_epochs, self.config.epochs):
            print(f"\nStage 2 - Epoch {epoch+1}/{self.config.epochs}")
            print("-" * 30)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_map = self.validate_epoch(epoch)
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_map > best_val_map
            if is_best:
                best_val_map = val_map
            
            self.save_checkpoint(epoch, val_map, is_best)
            
            # Save training history
            self.save_training_history()
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"Stage 2 Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val mAP: {val_map:.4f}")
            print(f"  Best mAP: {best_val_map:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Elapsed Time: {elapsed/3600:.2f}h")
        
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Two-stage training completed in {total_time/3600:.2f} hours")
        print(f"Final validation mAP: {best_val_map:.4f}")
        print(f"Stage 1 (heads): {heads_epochs} epochs")
        print(f"Stage 2 (all): {remaining_epochs} epochs")
        print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on Food Dataset')
    parser.add_argument('--config', type=str, default='food_4class', 
                       help='Config name')
    parser.add_argument('--data_root', type=str, 
                       default=str((Path(__file__).resolve().parent / 'datasets' / 'swiss_coco_7class').resolve()),
                       help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, 
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, 
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18','resnet50','resnet101','resnet152'],
                        help='Backbone architecture')
    
    args = parser.parse_args()
    
    # Load config using factory
    config = get_config(
        args.config,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone=args.backbone
    )
    
    # Create trainer
    trainer = FoodMaskRCNNTrainer(config)
    
    # Build model
    trainer.build_model()
    
    # Setup data loaders
    trainer.setup_data_loaders()
    
    # Setup optimizer
    trainer.setup_optimizer()
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.val_maps = checkpoint.get('val_maps', [])
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
