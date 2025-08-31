#!/usr/bin/env python3
"""
Configuration for Food Mask R-CNN Training
Based on the shapes training notebook configuration
"""

import os
from pathlib import Path
from food_dataset import FoodDatasetTransforms

class FoodMaskRCNNConfig:
    """Configuration for Food Mask R-CNN training"""
    
    def __init__(self, name="food_maskrcnn", **kwargs):
        # Model name
        self.name = name
        
        # Dataset configuration
        self.data_root = kwargs.get('data_root', '../data/merged/rte8_7class_food_dataset')
        self.class_names = kwargs.get('class_names', ["carb", "meat", "vegetable", "others"])
        self.num_classes = len(self.class_names) + 1  # +1 for background
        
        # Training configuration - RTX 2060 optimized
        self.epochs = kwargs.get('epochs', 30)
        self.heads_epochs = kwargs.get('heads_epochs', max(1, self.epochs // 3))  # 33% for heads
        self.batch_size = kwargs.get('batch_size', 1)  # Reduce for 6GB VRAM
        self.num_workers = 0  # Always 0 on Windows to prevent hanging
        
        # Model configuration - ResNet-18 backbone (memory optimized)
        self.backbone = kwargs.get('backbone', 'resnet18')
        self.pretrained_backbone = True
        self.pretrained = True
        
        # Image configuration - configurable for different GPUs
        self.image_min_size = kwargs.get('image_min_size', 320)
        self.image_max_size = kwargs.get('image_max_size', 480)
        
        # Optimizer configuration
        self.optimizer = kwargs.get('optimizer', 'SGD')
        self.learning_rate = kwargs.get('learning_rate', 0.005)
        self.momentum = kwargs.get('momentum', 0.9)
        self.weight_decay = kwargs.get('weight_decay', 0.0005)
        
        # Learning rate scheduler
        self.scheduler = kwargs.get('scheduler', 'step')
        self.lr_step_size = kwargs.get('lr_step_size', 16)
        self.lr_gamma = kwargs.get('lr_gamma', 0.1)
        
        # Training parameters
        self.print_freq = kwargs.get('print_freq', 10)
        self.save_freq = kwargs.get('save_freq', 5)
        
        # Paths
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 
                                       f'./checkpoints/{self.name}')
        self.log_dir = kwargs.get('log_dir', f'./logs/{self.name}')
        
        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Data transforms
        self.train_transforms = FoodDatasetTransforms(training=True)
        self.val_transforms = FoodDatasetTransforms(training=False)
        
        # Detection parameters (similar to shapes config)
        self.detection_min_confidence = 0.01  # Very low for early training
        self.detection_nms_threshold = 0.3
        self.detection_max_instances = 100
        
        # RPN parameters
        self.rpn_anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        self.rpn_aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.rpn_anchor_sizes)
        
        print(f"Configuration: {self.name}")
        print(f"  Classes: {self.num_classes} ({self.class_names})")
        print(f"  Data root: {self.data_root}")
        print(f"  Epochs: {self.epochs} (heads: {self.heads_epochs})")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Backbone: {self.backbone}")
    
    def display(self):
        """Display configuration like in the shapes notebook"""
        print("\nConfigurations:")
        print(f"NAME                           {self.name}")
        print(f"BACKBONE                       {self.backbone}")
        print(f"NUM_CLASSES                    {self.num_classes}")
        print(f"BATCH_SIZE                     {self.batch_size}")
        print(f"LEARNING_RATE                  {self.learning_rate}")
        print(f"MOMENTUM                       {self.momentum}")
        print(f"WEIGHT_DECAY                   {self.weight_decay}")
        print(f"EPOCHS                         {self.epochs}")
        print(f"HEADS_EPOCHS                   {self.heads_epochs}")
        print(f"IMAGE_MIN_SIZE                 {self.image_min_size}")
        print(f"IMAGE_MAX_SIZE                 {self.image_max_size}")
        print(f"DETECTION_MIN_CONFIDENCE       {self.detection_min_confidence}")
        print(f"DETECTION_NMS_THRESHOLD        {self.detection_nms_threshold}")
        print(f"DETECTION_MAX_INSTANCES        {self.detection_max_instances}")
        print(f"OPTIMIZER                      {self.optimizer}")
        print(f"SCHEDULER                      {self.scheduler}")
        print(f"DATA_ROOT                      {self.data_root}")
        print(f"CHECKPOINT_DIR                 {self.checkpoint_dir}")
        print(f"LOG_DIR                        {self.log_dir}")

class Food4ClassConfig(FoodMaskRCNNConfig):
    """Configuration for 4-class food segmentation"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('class_names', ["carb", "meat", "vegetable", "others"])
        super().__init__(name="food_4class", **kwargs)

class Food7ClassConfig(FoodMaskRCNNConfig):
    """Configuration for 7-class food segmentation"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('class_names', ["fruit", "vegetable", "carbohydrate", "protein", "dairy", "fat", "other"])
        super().__init__(name="food_7class", **kwargs)

class FoodRTXConfig(FoodMaskRCNNConfig):
    """Configuration optimized for RTX 2060 (6GB VRAM)"""
    
    def __init__(self, **kwargs):
        # RTX 2060 optimizations
        kwargs.setdefault('batch_size', 2)
        kwargs.setdefault('num_workers', 0)
        kwargs.setdefault('epochs', 40)
        kwargs.setdefault('heads_epochs', 8)
        kwargs.setdefault('learning_rate', 0.003)
        kwargs.setdefault('image_min_size', 416)
        kwargs.setdefault('image_max_size', 832)
        
        super().__init__(name="food_rtx2060", **kwargs)
        
        print("RTX 2060 optimizations applied:")
        print(f"  Reduced batch size: {self.batch_size}")
        print(f"  Reduced image size: {self.image_min_size}-{self.image_max_size}")
        print(f"  Adjusted epochs: {self.epochs}")

# Configuration factory
def get_config(config_name, **kwargs):
    """Get configuration by name"""
    configs = {
        'food_4class': Food4ClassConfig,
        'food_7class': Food7ClassConfig,
        'swiss_7class': Food7ClassConfig,  # Add swiss_7class alias
        'food_rtx': FoodRTXConfig,
        'default': FoodMaskRCNNConfig
    }
    
    config_class = configs.get(config_name, FoodMaskRCNNConfig)
    return config_class(**kwargs)

if __name__ == "__main__":
    # Test configurations
    print("Testing configurations...")
    
    config = Food4ClassConfig()
    config.display()
    
    print("\n" + "="*50)
    
    rtx_config = FoodRTXConfig()
    rtx_config.display()