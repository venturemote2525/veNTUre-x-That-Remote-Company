#!/usr/bin/env python3
"""
Small launcher to start Mask R-CNN training with a specific dataset path
(avoids Windows shell quoting issues with spaces in paths).
"""

from pathlib import Path
import sys

# Ensure we can import the training script and its utils
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

from train_maskrcnn_food import FoodMaskRCNNTrainer
from utils.config import get_config


def main():
    # Hardcode dataset path here to avoid quoting issues in cmd.exe
    # Update this if you want to switch datasets.
    dataset_root = r"D:\githib repo clones\veNTUre-x-That-Remote-Company\src\segmentation\mask_rcnn\datasets\Modified\7class_food_maskrcnn"

    config = get_config(
        'swiss_7class',
        data_root=dataset_root,
        epochs=25,
        backbone='resnet152'
    )

    trainer = FoodMaskRCNNTrainer(config)
    trainer.build_model()
    trainer.setup_data_loaders()
    trainer.setup_optimizer()
    trainer.train()


if __name__ == "__main__":
    main()

