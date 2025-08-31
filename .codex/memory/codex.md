# Codex Memory — Food Portion Size Classifier

This repository implements food instance segmentation (Mask R-CNN) with optional depth estimation (Depth Anything) to support portion analysis. Optimized for NVIDIA RTX 3060 (12GB).

## Summary

- Focus: 7‑class Swiss food instance segmentation (+ background)
- Primary model: Mask R‑CNN (typical backbones: ResNet‑18/50/152)
- Depth: Depth Anything integration for depth/3D cues
- Monitoring: CLI status, Streamlit dashboard, Tkinter GUI
- Artifacts: Logs and checkpoints tracked per configuration

## Key Entry Points

- Train segmentation: `python src/training/train_swiss_7class.py`
  - Schedule: ~70 epochs (23 heads‑only + 47 full fine‑tune)
  - Produces: checkpoints + `training_history.json`

- Depth pipeline: `python src/training/train_depth_anything_food.py`
  - Examples:
    - Dataset split: `--dataset-root data/swiss_coco_7class --output-dir outputs/depth_anything --split val`
    - Single image: `--single-image path/to/image.jpg --output-dir outputs/depth_anything`

- Status (CLI): `python scripts/show_status.py`
  - Shows latest logs/checkpoints, current/last epoch, loss, run status

- Streamlit monitor: `python ui/training_monitor.py`
  - Auto‑detects latest training log; plots train/val loss and mAP

- Tkinter monitor: `python gui_monitor/main_dashboard.py`
  - Real‑time GPU/util/temp/memory/power + training loss graphs; log browser

## Important Paths

- Logs (examples):
  - `src/training/logs/swiss_7class_resnet152/` (recent logs)
  - May also use `src/training/logs/swiss_7class_resnet50` and `logs/rtx3060_segmentation`

- Checkpoints (examples):
  - `src/training/checkpoints/swiss_7class_resnet50/`
  - Files: `best_checkpoint.pth`, `latest_checkpoint.pth`, `checkpoint_epoch_XXX.pth`, `training_history.json`

- Depth module: `src/models/depth_anything/` (TorchHub DINOv2, Streamlit demo, assets)

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

## Open Items

- `claude.md` not found in repo. If you provide it (or its path), I will add/merge its contents here or store it alongside this file.

