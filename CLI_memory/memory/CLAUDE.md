# Codex Memory — Food Nutrition Analysis System

This repository implements an integrated food nutrition analysis pipeline combining instance segmentation (Mask R-CNN), depth estimation (Depth Anything), and nutritional analysis for comprehensive food portion and calorie estimation. Optimized for NVIDIA RTX 3060 (12GB).

## Summary

- Focus: Complete nutrition analysis from food images
- Pipeline: Segmentation → Depth → Volume → Nutrition estimation
- Models: Mask R‑CNN (7-class Swiss food), Depth Anything, Nutrition database
- Integration: Support for specific food classification (friend's model)
- Interface: Unified Streamlit app with export capabilities
- Monitoring: CLI status, Streamlit dashboard, Tkinter GUI
- Artifacts: Logs, checkpoints, and UML architecture documentation

## Key Entry Points

### Nutrition Analysis Pipeline
- **Unified Nutrition App**: `python -m streamlit run src/nutrition_analysis/streamlit_nutrition_app.py`
  - Integrated segmentation + depth + nutrition analysis
  - Upload images → get calories, macros, volume estimates
  - Export results as JSON/CSV
  - Support for friend's specific food classifier

### 🐳 Docker Deployment (Recommended)
- **Quick Start**: `docker-compose up -d`
  - Complete containerized environment with all dependencies
  - Automatic model downloads and caching
  - Production-ready with health checks and monitoring
  - Access at: http://localhost:8501

- **Single Container**: `docker build -t nutrition-app . && docker run -p 8501:8501 nutrition-app`
  - Minimal deployment for testing or lightweight usage

### Training & Development
- Train segmentation: `python src/training/train_swiss_7class.py`
  - Schedule: ~70 epochs (23 heads‑only + 47 full fine‑tune)
  - Produces: checkpoints + `training_history.json`

#### Current Training Session (September 1, 2025)
- **Model**: ResNet-101 backbone (configurable: resnet18/50/101/152)
- **Command Template**: `python src/training/train_swiss_7class.py --backbone <resnet_version> --epochs <num_epochs>`
- **Current**: `python src/training/train_swiss_7class.py --backbone resnet101 --epochs 40`
- **Status**: Training in progress (40 epochs)
- **Config**: Batch size 6, 6 workers, 800px images (RTX 3060 optimized)
- **Log Location**: `src/training/logs/swiss_7class_<backbone>/training_YYYYMMDD_HHMMSS.log`
  - ResNet-101: `src/training/logs/swiss_7class_resnet101/`
  - ResNet-152: `src/training/logs/swiss_7class_resnet152/`

- Depth pipeline: `python src/training/train_depth_anything_food.py`
  - Examples:
    - Dataset split: `--dataset-root data/swiss_coco_7class --output-dir outputs/depth_anything --split val`
    - Single image: `--single-image path/to/image.jpg --output-dir outputs/depth_anything`

### Monitoring & Status
- Status (CLI): `python scripts/show_status.py`
  - Shows latest logs/checkpoints, current/last epoch, loss, run status

- Streamlit monitor: `python ui/streamlit/training_monitor.py`
  - Auto‑detects latest training log; plots train/val loss and mAP

- Tkinter monitor: `python ui/gui_monitor/main_dashboard.py`
  - Real‑time GPU/util/temp/memory/power + training loss graphs; log browser

## Important Paths

### Nutrition Analysis Module
- **Main pipeline**: `src/nutrition_analysis/`
  - `nutrition_pipeline.py` - Main integration class
  - `nutrition_database.py` - Food nutrition lookup (70+ foods)
  - `volume_calculator.py` - Enhanced 3D volume estimation
  - `specific_classifier_interface.py` - Interface for friend's model
  - `streamlit_nutrition_app.py` - Unified Streamlit interface
  - `pipeline_architecture.puml` - UML system architecture

### Training & Models
- Logs (examples):
  - `src/training/logs/swiss_7class_resnet152/` (recent logs)
  - May also use `src/training/logs/swiss_7class_resnet50` and `logs/rtx3060_segmentation`

- Checkpoints (examples):
  - `src/training/checkpoints/swiss_7class_resnet50/`
  - Files: `best_checkpoint.pth`, `latest_checkpoint.pth`, `checkpoint_epoch_XXX.pth`, `training_history.json`

- Models:
  - Segmentation: `src/models/pytorch_mask_rcnn/` (Mask R-CNN implementation)
  - Depth: `src/models/depth_anything/` (TorchHub DINOv2, Streamlit demo, assets)

## Dataset Notes

- Primary dataset: `data/swiss_coco_7class/` (COCO format; train/val)
- Additional: `data/classification/` (5/7/8‑class), `data/coco_5class_food_dataset/`, `data/merged/`

## Hardware & Performance

- Target GPU: RTX 3060 (12GB)
- Suggested settings: batch size 2–4; workers 2–4
- CUDA allocator (Windows example):
  - `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - Example run: `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python src/training/train_swiss_7class.py --batch-size 4`

## Claude Settings Found

- `.claude/settings.local.json` allows: `python:*`, `pip install:*`, `streamlit run:*`

## Collaboration Preferences (User Rules)

- Prefer edits over new files: Do not create new files unless no suitable existing file can be updated. If a new file is necessary, keep it minimal and justified.
- No random utility/checking scripts: Avoid adding ad‑hoc scripts for checks or helpers unless explicitly requested or essential to the task.
- No dummy data or placeholders: Do not use dummy variables, mock data, or placeholder scripts. Operate on real data/paths relevant to the project.
- Respect specified models: Do not switch to different models/architectures just to work around issues. If blocked, surface the blocker and ask before changing scope.
- Build the full product when asked: Do not deliver demos/prototypes when the request is for a complete implementation.
- Communicate blockers early: If something prevents completion, report it and request guidance rather than altering requirements.

## Nutrition Analysis Features

### Pipeline Capabilities
- **Food Detection**: 7-class Swiss food segmentation (fruit, vegetable, carbohydrate, protein, dairy, fat, other)
- **3D Volume Estimation**: Enhanced volume calculation using depth maps + segmentation masks
- **Nutritional Database**: 70+ food items with calories, macronutrients, densities
- **Specific Classification**: Extensible interface for friend's specific food classifier
- **Export & Analysis**: JSON/CSV export, detailed nutrition breakdowns, visualization charts

### Integration Points
- **Friend's Model Interface**: `SpecificClassifierInterface` allows seamless integration
- **Template Provided**: `FriendsSpecificClassifierTemplate` shows implementation pattern
- **Broad → Specific Mapping**: Falls back to broad classes when specific classification unavailable

### Output Format
```json
{
  "summary": {
    "total_calories": 450.2,
    "total_volume_ml": 280.5,
    "total_protein_g": 15.2,
    "total_carbs_g": 65.3,
    "total_fat_g": 12.1
  },
  "food_items": [
    {
      "broad_class": "carbohydrate",
      "specific_class": "rice",
      "volume_ml": 120.0,
      "calories": 180.5,
      "confidence": "high"
    }
  ]
}
```

## Architecture
- **UML Documentation**: Complete system architecture in `pipeline_architecture.puml`
- **Modular Design**: Separate components for segmentation, depth, volume, nutrition
- **Extensible**: Easy integration of new models and nutrition data

## 🐳 Docker Configuration

### Container Features
- **Multi-stage build**: Optimized for production with minimal image size
- **Security**: Non-root user, minimal attack surface
- **Caching**: Persistent model cache, automatic downloads
- **Health monitoring**: Built-in health checks for container orchestration
- **GPU support**: Optional NVIDIA Docker runtime for acceleration

### Deployment Options

#### Development
```bash
# Clone and start full stack
git clone <repository>
cd <repository>
docker-compose up -d

# Access nutrition analysis app
open http://localhost:8501
```

#### Production
```bash
# Build optimized image
docker build -t food-nutrition-analysis:production .

# Run with model volume mounts
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/src/training/checkpoints:/app/src/training/checkpoints:ro \
  -v nutrition_cache:/app/.cache \
  --name nutrition-analysis \
  food-nutrition-analysis:production
```

#### Cloud Deployment
- **Kubernetes**: Ready for K8s with health checks and resource limits
- **AWS ECS/Fargate**: Container-native deployment with auto-scaling
- **Google Cloud Run**: Serverless deployment with automatic scaling
- **Azure Container Instances**: Simple cloud container deployment

### Volume Configuration
- **Model checkpoints**: `/app/src/training/checkpoints` (read-only)
- **Model cache**: `/app/.cache` (persistent, ~2GB for models)
- **Outputs**: `/app/outputs` (analysis results)
- **Temporary**: `/app/temp` (image processing workspace)

### Environment Variables
- `DEPTH_ENCODER`: vitb|vits|vitl (model size)
- `CONFIDENCE_THRESHOLD`: 0.0-1.0 (detection confidence)
- `USE_SPECIFIC_CLASSIFIER`: true|false (friend's model)
- `DEVICE`: cuda|cpu (computation device)

### Integration as Backend Service
The Docker container provides HTTP endpoints that can be consumed by:
- **Mobile apps**: React Native, Flutter, iOS/Android
- **Web applications**: React, Vue, Angular frontends
- **API gateways**: AWS API Gateway, Azure API Management
- **Microservices**: Part of larger food/health applications


## Updates (Reference Scale)\n- New CLI: python reference_scale_pipeline/run.py --image path/to.jpg\n- Uses reference objects (credit card/spoon) to calibrate mm-per-pixel\n- Override sizes via eference_scale_pipeline/reference_objects.yaml\n- UML updated: src/nutrition_analysis/pipeline_architecture.puml includes Reference Scale module\n

### Repo Structure Update

## Structure (Updated)
- src/segmentation/mask_rcnn: Mask R-CNN code, training, datasets, checkpoints
- src/depth/depth_anything: Depth model code and training
- src/volume: Volume calculator
- src/reference: Reference-scale detection
- src/nutrition: Nutrition database
- src/specific_classification: Interface for specific food models
- src/pipeline: Orchestration pipeline, Streamlit app, docs, CLI

### Entry Points (Updated)
- App: `streamlit run src/pipeline/streamlit_app.py`
- CLI: `python src/pipeline/cli/reference_scale_run.py --image path/to.jpg`

### Docker Paths (Updated)
- CMD app: `src/pipeline/streamlit_app.py`
- Volumes:
  - `./src/segmentation/mask_rcnn/training/checkpoints:/app/src/segmentation/mask_rcnn/training/checkpoints:ro`
  - `./src/segmentation/mask_rcnn/datasets:/app/src/segmentation/mask_rcnn/datasets:ro`

## Training Paths (Updated)
- Segmentation datasets: src/segmentation/mask_rcnn/datasets/
- Segmentation logs: src/segmentation/mask_rcnn/training/logs/<model_name>/
- Segmentation checkpoints: src/segmentation/mask_rcnn/training/checkpoints/<model_name>/
- Segmentation exported models: src/segmentation/mask_rcnn/models/<model_name>/
- Depth logs: src/depth/training/logs/depth_anything_<encoder>/

