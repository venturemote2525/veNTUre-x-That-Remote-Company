#!/usr/bin/env python3
"""
YOLOv8 Utensil Detection Training

Trains a YOLO model to detect spoons and forks for reference scale measurement.
This script uses bounding box detection which is sufficient for scale calculation.
"""

import os
import sys
from pathlib import Path
import logging
import json
import shutil
from typing import Dict, List, Tuple

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class UtensilDetectionTrainer:
    """Trains YOLOv8 model for utensil detection."""
    
    def __init__(self, 
                 dataset_path: str = None,
                 model_size: str = 'yolov8n',
                 epochs: int = 50,
                 batch_size: int = 16,
                 img_size: int = 640,
                 device: str = None):
        """
        Initialize trainer.
        
        Args:
            dataset_path: Path to COCO format dataset
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            epochs: Training epochs
            batch_size: Batch size
            img_size: Input image size
            device: Device ('cuda', 'cpu', or None for auto)
        """
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.dataset_path = Path(dataset_path) if dataset_path else self._find_dataset()
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Output paths
        self.output_dir = Path(__file__).parent / "models"
        self.output_dir.mkdir(exist_ok=True)
        
        # Class mapping for utensils
        self.class_names = ['spoon', 'fork']
        
    def _find_dataset(self) -> Path:
        """Find utensil dataset in reference/datasets folder."""
        reference_dir = Path(__file__).parent
        dataset_dirs = [
            reference_dir / "datasets" / "unmodified",
            reference_dir / "datasets" / "modified",
        ]
        
        for dataset_dir in dataset_dirs:
            if dataset_dir.exists():
                self.logger.info(f"Checking dataset dir: {dataset_dir}")
                # Look for COCO annotations in subdirectories
                ann_files = list(dataset_dir.glob("**/*.json"))
                if ann_files:
                    self.logger.info(f"Found dataset: {dataset_dir}")
                    return dataset_dir
                    
        raise FileNotFoundError("No utensil dataset found in reference/datasets/")
    
    def prepare_yolo_dataset(self) -> Path:
        """Convert COCO format to YOLO format."""
        self.logger.info("Preparing YOLO dataset...")
        
        yolo_dir = self.output_dir / "yolo_dataset"
        yolo_dir.mkdir(exist_ok=True)
        
        # Create YOLO directory structure
        for split in ['train', 'val']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Find annotation files in subdirectories
        ann_files = list(self.dataset_path.glob("*/*.json"))
        if not ann_files:
            raise FileNotFoundError(f"No COCO annotation files found in {self.dataset_path}")
        
        train_ann = None
        val_ann = None
        
        for ann_file in ann_files:
            if 'train' in ann_file.parent.name:
                train_ann = ann_file
            elif 'valid' in ann_file.parent.name or 'val' in ann_file.parent.name:
                val_ann = ann_file
        
        # Use first annotation file if train/val not found
        if not train_ann and ann_files:
            train_ann = ann_files[0]
            self.logger.info(f"Using single annotation file: {train_ann}")
        
        # Convert annotations
        total_images = 0
        if train_ann:
            train_count = self._convert_coco_to_yolo(train_ann, yolo_dir / 'train')
            total_images += train_count
            
        if val_ann:
            val_count = self._convert_coco_to_yolo(val_ann, yolo_dir / 'val')
            total_images += val_count
        elif train_ann:
            # Split training data for validation
            val_count = self._split_for_validation(yolo_dir, split_ratio=0.2)
            total_images += val_count
        
        # Create dataset.yaml
        dataset_yaml = yolo_dir / 'dataset.yaml'
        yaml_content = {
            'path': str(yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(yaml_content, f)
        
        self.logger.info(f"Prepared YOLO dataset with {total_images} total images")
        return dataset_yaml
        
    def _convert_coco_to_yolo(self, coco_file: Path, output_dir: Path) -> int:
        """Convert COCO annotations to YOLO format."""
        import json
        from pycocotools.coco import COCO
        
        self.logger.info(f"Converting {coco_file} to YOLO format...")
        
        # Load COCO dataset
        coco = COCO(str(coco_file))
        
        # Get image and annotation info
        image_ids = list(coco.imgs.keys())
        converted_count = 0
        
        for image_id in image_ids:
            img_info = coco.imgs[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Find image file
            img_path = None
            # Look in the same directory as the annotation file
            ann_dir = coco_file.parent
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = ann_dir / img_filename
                if not potential_path.exists():
                    # Try without extension
                    base_name = Path(img_filename).stem
                    potential_path = ann_dir / f"{base_name}{ext}"
                
                if potential_path.exists():
                    img_path = potential_path
                    break
            
            if not img_path or not img_path.exists():
                continue
            
            # Copy image
            output_img_path = output_dir / 'images' / img_filename
            if not output_img_path.exists():
                shutil.copy2(img_path, output_img_path)
            
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(ann_ids)
            
            # Convert annotations to YOLO format
            yolo_labels = []
            for ann in anns:
                if 'bbox' not in ann:
                    continue
                    
                # Get category (map to our utensil classes)
                cat_info = coco.cats.get(ann['category_id'], {})
                cat_name = cat_info.get('name', '').lower()
                
                # Map to our classes
                class_idx = None
                if 'spoon' in cat_name:
                    class_idx = 0  # spoon
                elif 'fork' in cat_name:
                    class_idx = 1  # fork
                
                # Debug: print category mapping
                if class_idx is None and cat_name:
                    print(f"Unknown category: {cat_name} (ID: {ann['category_id']})")
                
                if class_idx is None:
                    continue
                
                # Convert bbox to YOLO format (center_x, center_y, width, height) normalized
                x, y, w, h = ann['bbox']
                center_x = (x + w/2) / img_width
                center_y = (y + h/2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                yolo_labels.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            # Save YOLO label file
            if yolo_labels:
                label_filename = Path(img_filename).stem + '.txt'
                label_path = output_dir / 'labels' / label_filename
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                converted_count += 1
        
        self.logger.info(f"Converted {converted_count} images from {coco_file}")
        return converted_count
        
    def _split_for_validation(self, yolo_dir: Path, split_ratio: float = 0.2) -> int:
        """Split training data for validation."""
        import random
        
        train_images_dir = yolo_dir / 'train' / 'images'
        train_labels_dir = yolo_dir / 'train' / 'labels'
        val_images_dir = yolo_dir / 'val' / 'images'
        val_labels_dir = yolo_dir / 'val' / 'labels'
        
        # Get all training images
        image_files = list(train_images_dir.glob('*'))
        random.shuffle(image_files)
        
        # Calculate split
        val_count = int(len(image_files) * split_ratio)
        val_images = image_files[:val_count]
        
        # Move validation images and labels
        moved_count = 0
        for img_file in val_images:
            label_file = train_labels_dir / f"{img_file.stem}.txt"
            
            # Move image
            val_img_path = val_images_dir / img_file.name
            shutil.move(img_file, val_img_path)
            
            # Move label if exists
            if label_file.exists():
                val_label_path = val_labels_dir / label_file.name
                shutil.move(label_file, val_label_path)
            
            moved_count += 1
        
        self.logger.info(f"Moved {moved_count} images to validation set")
        return moved_count
        
    def train(self):
        """Train YOLOv8 model for utensil detection."""
        self.logger.info("Starting YOLOv8 utensil detection training...")
        
        try:
            # Prepare dataset
            dataset_yaml = self.prepare_yolo_dataset()
            
            # Initialize YOLOv8 model
            model_name = f"yolov8{self.model_size[-1]}.pt"  # e.g., yolov8n.pt
            model = YOLO(model_name)
            
            # Training arguments
            train_args = {
                'data': str(dataset_yaml),
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'device': self.device,
                'project': str(self.output_dir),
                'name': 'utensil_detection',
                'exist_ok': True,
                'patience': 10,
                'save_period': 10,
                'optimizer': 'SGD',
                'lr0': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
            }
            
            # Start training
            self.logger.info(f"Training with args: {train_args}")
            results = model.train(**train_args)
            
            # Save final model to reference folder
            final_model_path = self.output_dir / 'utensil_detector_yolo.pt'
            best_model_path = self.output_dir / 'utensil_detection' / 'weights' / 'best.pt'
            
            if best_model_path.exists():
                shutil.copy2(best_model_path, final_model_path)
                self.logger.info(f"Saved final model to: {final_model_path}")
            
            # Print training summary
            self.logger.info("Training completed!")
            self.logger.info(f"Model saved to: {final_model_path}")
            self.logger.info(f"Results: {results}")
            
            return final_model_path
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 utensil detector')
    parser.add_argument('--dataset', type=str, help='Path to COCO dataset')
    parser.add_argument('--model', type=str, default='yolov8n', 
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = UtensilDetectionTrainer(
        dataset_path=args.dataset,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device
    )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()