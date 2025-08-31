#!/usr/bin/env python3
"""
Resume Stage 2 (full fine-tuning) for Swiss 7-class Mask R-CNN training.

Purpose:
- Skip heads-only training and resume fine-tuning all layers from the latest checkpoint.

Usage:
- From repo root or src/training, run:
  python src/training/resume_stage2_swiss_7class.py \
    --checkpoint src/training/checkpoints/swiss_7class_resnet18/latest_checkpoint.pth \
    --epochs 70 \
    --data_root data/swiss_coco_7class

Notes:
- Defaults match train_swiss_7class.py and current repo layout.
- Creates a new log file under the existing logs directory.
"""

import sys
from pathlib import Path
import argparse
import torch

# Add pytorch_mask_rcnn to path (same as train_swiss_7class.py)
here = Path(__file__).resolve().parent
maskrcnn_path = here.parent / "models" / "pytorch_mask_rcnn"
sys.path.append(str(maskrcnn_path))
sys.path.append(str(maskrcnn_path / "utils"))

from utils.config import FoodMaskRCNNConfig
from train_maskrcnn_food import FoodMaskRCNNTrainer
import torch.optim as optim


def main():
    parser = argparse.ArgumentParser(description="Resume stage-2 fine-tuning for Swiss 7-class")
    parser.add_argument("--checkpoint", type=str,
                        default=str(here / "checkpoints" / "swiss_7class_resnet18" / "latest_checkpoint.pth"),
                        help="Path to latest checkpoint .pth")
    parser.add_argument("--epochs", type=int, default=70, help="Total epochs target (same as initial plan)")
    parser.add_argument("--data_root", type=str,
                        default=str(here.parent.parent / "data" / "swiss_coco_7class"),
                        help="Dataset root path")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Swiss 7-class label set (consistent with prior training)
    swiss_7_classes = ['fruit', 'vegetable', 'carbohydrate', 'protein', 'dairy', 'fat', 'other']

    # Build config matching the ongoing experiment name and settings
    config = FoodMaskRCNNConfig(
        name='swiss_7class_resnet18',
        data_root=str(Path(args.data_root)),
        backbone='resnet18',
        epochs=args.epochs,
        batch_size=1,
        learning_rate=0.001,
        class_names=swiss_7_classes,
        num_workers=0,
        # Keep outputs consistent with existing run locations
        checkpoint_dir=str(here / 'checkpoints' / 'swiss_7class_resnet18'),
        log_dir=str(here / 'logs' / 'swiss_7class_resnet18')
    )

    # Initialize trainer pipeline
    trainer = FoodMaskRCNNTrainer(config)
    trainer.build_model()
    trainer.setup_data_loaders()
    trainer.setup_optimizer()

    # Load checkpoint state
    print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Determine resume point
    last_epoch = int(checkpoint.get('epoch', -1))
    start_epoch = last_epoch + 1  # next epoch to run
    print(f"Last completed epoch in checkpoint: {last_epoch}")
    print(f"Resuming Stage 2 from epoch: {start_epoch}")

    # Stage 2 setup: unfreeze all layers and lower LR
    trainer.unfreeze_all_layers()
    finetune_lr = config.learning_rate / 10
    for pg in trainer.optimizer.param_groups:
        pg['lr'] = finetune_lr

    # Reset scheduler for the remaining epochs only
    remaining_epochs = max(0, config.epochs - start_epoch)
    if config.scheduler == 'step':
        trainer.scheduler = optim.lr_scheduler.StepLR(
            trainer.optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
    elif config.scheduler == 'cosine':
        trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=remaining_epochs
        )

    # Carry over history (optional)
    trainer.train_losses = checkpoint.get('train_losses', [])
    trainer.val_losses = checkpoint.get('val_losses', [])
    trainer.val_maps = checkpoint.get('val_maps', [])

    # Run Stage 2 loop only
    import time
    best_val_map = max(trainer.val_maps) if trainer.val_maps else 0.0
    stage2_start = time.time()
    for epoch in range(start_epoch, config.epochs):
        print(f"\nStage 2 - Epoch {epoch+1}/{config.epochs}")
        print("-" * 30)

        train_loss = trainer.train_epoch(epoch)
        val_loss, val_map = trainer.validate_epoch(epoch)

        if trainer.scheduler:
            trainer.scheduler.step()

        is_best = val_map > best_val_map
        if is_best:
            best_val_map = val_map

        trainer.save_checkpoint(epoch, val_map, is_best)
        trainer.save_training_history()

        current_lr = trainer.optimizer.param_groups[0]['lr']
        elapsed = time.time() - stage2_start
        print("Stage 2 Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val mAP: {val_map:.4f}")
        print(f"  Best mAP: {best_val_map:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Elapsed Time: {elapsed/3600:.2f}h")

    print("\nStage 2 fine-tuning complete.")


if __name__ == "__main__":
    main()
