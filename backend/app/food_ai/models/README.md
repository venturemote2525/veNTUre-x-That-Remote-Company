# AI Models for Food Nutrition Analysis System

This directory contains all AI models used by the food nutrition analysis system.

## Model Structure

```
models/
├── segmentation/           # Food segmentation models
├── classification/         # Food classification models
├── utensil_detection/     # Utensil detection for scale
├── depth_estimation/      # Depth estimation models
└── README.md              # This file
```

## Models Overview

### 1. Food Segmentation (7-Class)
**Location:** `segmentation/`
- **Primary:** `segformer_best_dice.pth` (180.4 MB)
- **Backup:** `segformer_final.pth`
- **Architecture:** SegFormer with MiT-B3 backbone
- **Classes:** fruit, vegetable, carbohydrate, protein, dairy, fat, other
- **Input:** RGB images (512x512)
- **Framework:** PyTorch + Transformers

### 2. Food Classification (244-Class)
**Location:** `classification/`
- **File:** `vit_small.onnx` (84.0 MB)
- **Architecture:** Vision Transformer Small
- **Classes:** 244 specific food types
- **Input:** RGB images (224x224)
- **Framework:** ONNX Runtime

### 3. Utensil Detection (2-Class)
**Location:** `utensil_detection/`
- **Primary:** `utensil_detector_converted.onnx` (11.7 MB) - **RECOMMENDED**
- **Backup:** `utensil_detector_yolo.pt` (6.0 MB)
- **Architecture:** YOLO v8
- **Classes:** spoon (18cm), fork (20cm)
- **Input:** RGB images (640x640)
- **Framework:** ONNX Runtime / PyTorch
- **Purpose:** Reference scaling for volume calculation

### 4. Depth Estimation
**Location:** `depth_estimation/`
- **File:** `midas_small_depth.pth` (81.8 MB)
- **Architecture:** MiDaS Small
- **Purpose:** Enhanced volume calculation from depth maps
- **Input:** RGB images
- **Framework:** PyTorch

## Model Usage Priority

### Food Analysis Pipeline:
1. **Segmentation:** PyTorch SegFormer (`segformer_best_dice.pth`)
2. **Classification:** ONNX ViT (`vit_small.onnx`)
3. **Utensil Detection:** ONNX YOLO (`utensil_detector_converted.onnx`)
4. **Depth Estimation:** PyTorch MiDaS (`midas_small_depth.pth`)

## Model Performance

### Expected Analysis Results (per meal):
- **Food Items:** 2-4 detected classes
- **Total Calories:** 200-1200 kcal (realistic meal portions)
- **Processing Time:** ~5-15 seconds on RTX 3060
- **Scale Accuracy:** ±10% with utensil detection

### Hardware Requirements:
- **GPU:** NVIDIA RTX 3060 (12GB) or equivalent
- **RAM:** 8GB+ system RAM
- **Storage:** ~400MB for all models
- **CUDA:** 11.8+ for PyTorch models

## Integration Points

### API Service:
- Models loaded in `src/api/food_analysis_service.py`
- Fallback mechanism: ONNX → PyTorch
- Device auto-detection: CUDA → CPU

### Streamlit App:
- Direct model loading for real-time analysis
- UI integration in `ui/nutrition_analysis_app.py`

## Git LFS Models (Included in Repository)

The following models are stored using Git LFS and included in the repository:

### Small-Medium Models (<100MB)
- **vit_food_classifier.onnx** (84MB) - Food classification
- **utensil_detector_converted.onnx** (43MB) - Utensil detection
- **utensil_detector_converted_12MB_backup.onnx** (12MB) - Backup utensil detector
- **yolo_utensil_detector_43MB.onnx** (43MB) - Alternative utensil detector

### Excluded Models (>100MB - GitHub LFS Limit)
These models exceed GitHub's 100MB LFS limit and must be downloaded separately:

- **segformer_food_segmentation.onnx** (181MB) - Food segmentation
- **depth_anything_vitb.onnx** (371MB) - Depth estimation

## Model Updates (Latest)

### October 2025:
- ✅ Implemented Git LFS for model storage
- ✅ Excluded largest models (>100MB) due to GitHub limits
- ✅ Updated model documentation
- ✅ Added fallback mechanisms for missing models
- ✅ Optimized repository size while maintaining functionality

### September 2025:
- ✅ Converted utensil detector to ONNX for consistency
- ✅ Verified 2-class utensil detection (spoon, fork)
- ✅ Optimized for realistic portion sizes
- ✅ Fixed scale calculation for accurate volume estimation
- ✅ Removed broken ONNX models (saved 219MB space)

### Model Validation:
All models tested with `testdata/Screenshot 2025-09-04 112203.png`:
- ✅ Segmentation: 3 food classes detected
- ✅ Classification: Specific food identification working
- ✅ Utensil Detection: Fork/spoon scale detection
- ✅ Realistic Results: 400-800 kcal for wrap/flatbread meal