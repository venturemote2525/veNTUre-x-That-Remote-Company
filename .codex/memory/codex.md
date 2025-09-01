# Codex Memory — Food Nutrition Analysis System

Integrated food nutrition analysis pipeline combining instance segmentation (Mask R-CNN), depth estimation (Depth Anything), 3D volume estimation, and nutrition lookup. Optimized for NVIDIA RTX 3060 (12GB).

## Summary

- Focus: Complete nutrition analysis from food images
- Pipeline: Segmentation → Depth → Volume → Nutrition
- Models: Mask R-CNN (Swiss 7-class), Depth Anything, nutrition database
- Integration: Optional specific-food classifier via interface
- Interfaces: Unified Streamlit app; CLI/Streamlit/Tkinter monitoring
- Artifacts: Logs, checkpoints, UML architecture

## Key Entry Points

- Unified nutrition app: `python -m streamlit run src/nutrition_analysis/streamlit_nutrition_app.py`
  - Upload images → get calories, macros, volume; export JSON/CSV
  - Supports optional friend’s specific classifier

- Train segmentation: `python src/training/train_swiss_7class.py`
  - Typical: ~70 epochs (23 heads-only + 47 full fine-tune)
  - Produces: checkpoints + `training_history.json`

- Depth pipeline: `python src/training/train_depth_anything_food.py`
  - Examples:
    - Dataset split: `--dataset-root data/swiss_coco_7class --output-dir outputs/depth_anything --split val`
    - Single image: `--single-image path/to/image.jpg --output-dir outputs/depth_anything`

- Status (CLI): `python scripts/show_status.py`
  - Shows latest logs/checkpoints, epoch, loss, run status

- Streamlit monitor: `python ui/streamlit/training_monitor.py`
- Tkinter monitor: `python ui/gui_monitor/main_dashboard.py`

### New: Reference-Scale CLI
- `python reference_scale_pipeline/run.py --image path/to.jpg`
  - Calibrates mm-per-pixel from a reference object (credit card/spoon) and runs analysis
  - Optional sizes override: `reference_scale_pipeline/reference_objects.yaml`

## Important Paths

- Nutrition module: `src/nutrition_analysis/`
  - `nutrition_pipeline.py`, `nutrition_database.py`, `volume_calculator.py`
  - `specific_classifier_interface.py`, `streamlit_nutrition_app.py`
  - `pipeline_architecture.puml`

- Logs: `src/training/logs/swiss_7class_<backbone>/`
- Checkpoints: `src/training/checkpoints/swiss_7class_<backbone>/`
- Segmentation model: `src/models/pytorch_mask_rcnn/`
- Depth model: `src/models/depth_anything/`

### New: Reference-Scale Pipeline
- Folder: `reference_scale_pipeline/`
  - `run.py`: CLI to run pipeline with reference-scale enabled
  - `reference_objects.yaml`: known object sizes (mm)

## Architecture (Updated)
- UML: `src/nutrition_analysis/pipeline_architecture.puml` now includes Reference Scale module

## Dataset Notes

- Primary: `data/swiss_coco_7class/` (COCO format; train/val)
- Additional: `data/classification/`, `data/coco_5class_food_dataset/`, `data/merged/`

## Hardware & Performance

- Target GPU: RTX 3060 (12GB)
- Suggested: batch size 2–4; workers 2–4
- CUDA allocator (Windows):
  - `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - Example: `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python src/training/train_swiss_7class.py --batch-size 4`

## Docker

- Quick start: `docker-compose up -d` (serves at http://localhost:8501)
- Single container: `docker build -t nutrition-app . && docker run -p 8501:8501 nutrition-app`
- Volumes: checkpoints `/app/src/training/checkpoints` (ro), cache `/app/.cache`, outputs `/app/outputs`, temp `/app/temp`
- Env vars: `DEPTH_ENCODER`, `CONFIDENCE_THRESHOLD`, `USE_SPECIFIC_CLASSIFIER`, `DEVICE`

## Claude Settings

- `.claude/settings.local.json` allows: `python:*`, `pip install:*`, `streamlit run:*`

## Collaboration Preferences (User Rules)

- Prefer edits over new files; keep new files minimal when necessary.
- No ad-hoc utility/check scripts unless requested/essential.
- No dummy data/placeholders; operate on real project paths/data.
- Don’t switch models/architectures without approval; surface blockers.
- Deliver full product when asked; avoid partial demos.
- Communicate blockers early.

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

