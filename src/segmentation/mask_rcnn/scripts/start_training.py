#!/usr/bin/env python3
"""
Start Mask R-CNN training on 7-class food dataset
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from utils.config import Food7ClassConfig, FoodRTXConfig
from train_maskrcnn_food import FoodMaskRCNNTrainer

def main():
    print("Starting Mask R-CNN Food Segmentation Training")
    print("=" * 60)
    
    # Use the COCO format 5-class dataset - already has all 3 datasets combined!
    data_root = "../../data/coco_5class_food_dataset"
    
    if Path(data_root).exists():
        print(f"Using merged 5-class dataset for SEGMENTATION training: {data_root}")
        print("This dataset contains ALL 3 initial datasets merged together!")
        class_names = ["carb", "meat", "vegetable", "fruit", "others"]
    else:
        print(f"Merged dataset not found at {data_root}")
        print("Available datasets:")
        for p in Path("../data/merged").glob("*"):
            if p.is_dir():
                print(f"  - {p}")
        return
    
    # Create configuration optimized for RTX 2060
    config = FoodRTXConfig(
        data_root=data_root,
        class_names=class_names,
        epochs=30,
        heads_epochs=6,
        batch_size=2,
        learning_rate=0.003
    )
    
    config.display()
    
    # Create trainer
    trainer = FoodMaskRCNNTrainer(config)
    
    try:
        # Build model
        print("\nBuilding Mask R-CNN with ResNet-50...")
        trainer.build_model()
        
        # Setup data loaders
        print("Setting up data loaders...")
        trainer.setup_data_loaders()
        
        # Setup optimizer
        print("Setting up optimizer and scheduler...")
        trainer.setup_optimizer()
        
        # Start training
        print("Starting two-stage training...")
        trainer.train()
        
        print("\nTraining completed successfully!")
        print(f"Checkpoints saved to: {config.checkpoint_dir}")
        print(f"Training history: {config.checkpoint_dir}/training_history.json")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Try with smaller configuration
        print("\nTrying with reduced settings...")
        
        config_small = FoodRTXConfig(
            data_root=data_root,
            class_names=class_names,
            epochs=20,
            heads_epochs=4,
            batch_size=1,  # Even smaller batch
            learning_rate=0.001
        )
        config_small.name = "food_rtx2060_small"
        
        trainer_small = FoodMaskRCNNTrainer(config_small)
        trainer_small.build_model()
        trainer_small.setup_data_loaders()
        trainer_small.setup_optimizer()
        trainer_small.train()

if __name__ == "__main__":
    main()