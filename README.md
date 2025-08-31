# Food Segmentation Project

A comprehensive deep learning project focused on **food segmentation** using Mask R-CNN and supporting depth estimation. The project performs instance segmentation of food items to identify and segment different food categories.

## Project Structure

### 📁 Main Directories

- **`src/`** - Source code organized by functionality
  - `training/` - Training scripts including main `train_swiss_7class.py`
  - `models/pytorch_mask_rcnn/` - Mask R-CNN implementation and utilities
  - `models/depth_anything/` - Depth estimation model integration
  - `utils/` - Utility functions and helpers

- **`data/`** - All datasets and annotations
  - `swiss_coco_7class/` - Swiss 7-class food segmentation dataset (COCO format)
  - `classification/` - Classification datasets (5, 7, 8-class variants)
  - `coco_5class_food_dataset/` - 5-class COCO format dataset
  - `merged/` - Combined and processed datasets

- **`scripts/`** - Utility and management scripts
  - `show_status.py` - Training status and progress monitoring
  - `manage_logs.py` - Log file management and organization

- **`logs/`** - Training logs organized by model configuration
  - `swiss_7class_resnet18/` - Mask R-CNN training logs

- **`checkpoints/`** - Model checkpoints and training states
  - `swiss_7class_resnet18/` - Saved model checkpoints and training history

## Models Supported

### 1. Mask R-CNN (Primary Focus)
- **Classes**: 7 Swiss food classes (fruit, vegetable, carbohydrate, protein, dairy, fat, other)
- **Backbone**: ResNet-18 (memory optimized for RTX 2060)
- **Features**: Instance segmentation with bounding boxes and masks
- **Training**: 70 epochs (23 heads-only + 47 full fine-tuning)
- **Dataset**: Swiss COCO 7-class food segmentation dataset

### 2. Depth Anything Integration
- **Purpose**: Depth estimation for portion analysis
- **Models**: Supports multiple Depth Anything variants
- **Integration**: Works with food segmentation pipeline
- **Applications**: 3D food analysis and portion estimation

## Quick Start

### Training Models

1. **Swiss 7-Class Food Segmentation (Primary)**:
   ```bash
   python src/training/train_swiss_7class.py
   ```

2. **Depth Estimation Pipeline**:
   ```bash
   # Dataset processing
   python src/training/train_depth_anything_food.py --dataset-root data/swiss_coco_7class --output-dir outputs/depth_anything --split val
   
   # Single image processing
   python src/training/train_depth_anything_food.py --single-image path/to/image.jpg --output-dir outputs/depth_anything
   ```

### Monitoring Training

Check training status and progress:
```bash
python scripts/show_status.py
```

Manage log files:
```bash
python scripts/manage_logs.py --create-links
```

## Dataset Information

### Swiss 7-Class Food Dataset (Primary)
- **Classes**: fruit, vegetable, carbohydrate, protein, dairy, fat, other (+ background)
- **Format**: COCO format with instance segmentation annotations
- **Location**: `data/swiss_coco_7class/` with `train/` and `val/` subfolders
- **Task**: Instance segmentation of food items for portion analysis

### Additional Datasets
- **Classification variants**: 5, 7, 8-class food datasets in `data/classification/`
- **COCO 5-class**: Alternative dataset format in `data/coco_5class_food_dataset/`
- **Merged datasets**: Combined datasets in `data/merged/`

## Memory Optimization

Optimized for training on RTX 3060 (12GB VRAM):

- **ResNet-18 backbone** for efficient training (can upgrade to ResNet-50 with 12GB VRAM)
- **Batch size 2-4** recommended for 12GB VRAM (increased from batch size 1)
- **Increased workers** (num_workers=2-4) for faster data loading
- **CUDA memory management** with expandable segments for stability

For maximum performance with RTX 3060:
```bash
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python src/training/train_swiss_7class.py --batch-size 4
```

## Features

- ✅ **Food instance segmentation** with Mask R-CNN
- ✅ **Depth estimation integration** for portion analysis
- ✅ **Memory-optimized training** for RTX 2060
- ✅ **Comprehensive logging** and checkpoint management
- ✅ **Training monitoring utilities** for progress tracking
- ✅ **Modular architecture** with organized source code structure

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- NVIDIA RTX 3060 or similar GPU (12GB VRAM)
- CUDA 12.1+ support
- Latest NVIDIA drivers (536.25+)

## Training Process

The main training script `src/training/train_swiss_7class.py` implements:
1. **23 epochs** of heads-only training (freeze backbone)
2. **47 epochs** of full model fine-tuning
3. **Automatic checkpointing** and training history logging
4. **Progress monitoring** through logging system

## Contributing

This is a research project focused on food segmentation and portion analysis using deep learning. The architecture is designed for systematic food portion estimation through instance segmentation and depth analysis.