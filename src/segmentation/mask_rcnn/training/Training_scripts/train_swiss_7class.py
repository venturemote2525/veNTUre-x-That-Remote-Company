#!/usr/bin/env python3
"""
Train Mask R-CNN on Swiss Food 7-Class Dataset - RTX 3060 High-Performance

RTX 3060 Optimized Defaults:
- ResNet-50 backbone (better accuracy with 12GB VRAM)
- Batch size 3 (balanced performance/memory)  
- 4 workers (fast data loading)

Usage examples:
  # RTX 3060 optimized defaults (ResNet-50, batch=3, workers=4)
  python src/training/train_swiss_7class.py

  # 2-epoch training session (quick test for RTX 3060)
  python src/training/train_swiss_7class.py --epochs 2

  # maximum performance (may use ~10-11GB VRAM)
  python src/training/train_swiss_7class.py --batch-size 4 --num-workers 6

  # conservative settings if memory issues
  python src/training/train_swiss_7class.py --batch-size 2 --num-workers 2 --backbone resnet18
"""

import sys
import os
from pathlib import Path
import argparse

# Add mask_rcnn module root to path (src/segmentation/mask_rcnn)
MASKRCNN_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(MASKRCNN_ROOT))
sys.path.append(str(MASKRCNN_ROOT / "utils"))

from utils.config import FoodMaskRCNNConfig
from train_maskrcnn_food import FoodMaskRCNNTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on Swiss Food 7-Class Dataset")
    parser.add_argument("--epochs", type=int, help="Total training epochs (default: 2)")
    parser.add_argument("--data-root", type=str, help="Path to swiss_coco_7class dataset root")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size (default: 6 for RTX 3060)")
    parser.add_argument("--num-workers", type=int, default=6, help="Number of data loading workers (default: 6)")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50", "resnet101", "resnet152"], help="Backbone architecture (default: resnet50)")
    parser.add_argument("--image-size", type=int, default=800, help="Image size for training (default: 800 for max VRAM usage)")
    parser.add_argument("--stage2-batch-size", type=int, help="Reduced batch size for Stage 2 full training (auto-calculated if not specified)")
    parser.add_argument("--time-limit", type=int, default=0, help="Time limit in hours (0 = no limit)")
    args = parser.parse_args()

    # Use only classes that exist in dataset annotations [1,2,3,7] -> [1,2,3,4]
    swiss_7_classes = ['carb', 'meat', 'vegetable', 'others']
    
    # Resolve dataset path
    default_data_root = MASKRCNN_ROOT / "datasets" / "swiss_coco_7class"
    data_root = Path(args.data_root) if args.data_root else default_data_root

    # Choose epochs
    total_epochs = int(args.epochs) if args.epochs else 2

    # Create config with optimized settings for RTX 3060
    config_name = f'swiss_7class_{args.backbone}'
    config = FoodMaskRCNNConfig(
        name=config_name,
        data_root=str(data_root),
        backbone=args.backbone,  # Configurable backbone
        epochs=total_epochs,
        batch_size=args.batch_size,  # Improved batch size for RTX 3060
        learning_rate=0.001,
        class_names=swiss_7_classes,
        num_workers=args.num_workers,  # Configurable workers
        image_min_size=args.image_size,  # High resolution for max VRAM usage
        image_max_size=args.image_size + 200,  # Max size for high quality
        checkpoint_dir=str(MASKRCNN_ROOT / 'training' / 'checkpoints' / config_name),
        log_dir=str(MASKRCNN_ROOT / 'training' / 'logs' / config_name),
        model_dir=str(MASKRCNN_ROOT / 'models' / config_name)
    )
    
    print(f"\n=== RTX 3060 2-Stage Mask R-CNN Training ===")
    print(f"Backbone: {args.backbone} (ResNet-50 for better accuracy)")
    print(f"Batch Size: {args.batch_size} (optimized for 12GB VRAM)")
    print(f"Workers: {args.num_workers} (parallel data loading)")
    print(f"Total Epochs: {total_epochs}")
    print(f"Training Type: 2-Stage (Heads-only + Full fine-tuning)")
    if args.time_limit > 0:
        print(f"Time Limit: {args.time_limit} hours")
    print(f"RTX 3060 12GB VRAM: Maximum performance settings")
    
    print(f"\n7 Food Classes: {swiss_7_classes}")
    print(f"Total classes (including background): {config.num_classes}")
    
    # Create trainer
    trainer = FoodMaskRCNNTrainer(config)
    
    # Build model
    trainer.build_model()
    
    # Setup data loaders
    trainer.setup_data_loaders()
    
    # Setup optimizer
    trainer.setup_optimizer()
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
