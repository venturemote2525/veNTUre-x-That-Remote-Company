#!/usr/bin/env python3
"""
Model configuration for food nutrition analysis system
Defines paths and settings for all AI models
"""

from pathlib import Path

# Base paths
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

# Model configurations
MODEL_CONFIG = {
    # Food segmentation (7-class)
    "segmentation": {
        "primary_model": MODELS_DIR / "segmentation" / "segformer_best_dice.pth",
        "backup_model": MODELS_DIR / "segmentation" / "segformer_final.pth",
        "architecture": "SegformerForSemanticSegmentation",
        "pretrained": "nvidia/mit-b3",
        "input_size": 512,
        "classes": [
            'background', 'fruit', 'vegetable', 'carbohydrate',
            'protein', 'dairy', 'fat', 'other'
        ],
        "framework": "pytorch",
        "size_mb": 181
    },

    # Food classification (244-class)
    "classification": {
        "primary_model": MODELS_DIR / "classification" / "vit_small.onnx",
        "architecture": "ViT-Small",
        "input_size": 224,
        "classes": 244,  # Specific food classes
        "framework": "onnx",
        "size_mb": 85
    },

    # Utensil detection (2-class)
    "utensil_detection": {
        "primary_model": MODELS_DIR / "utensil_detection" / "utensil_detector_converted.onnx",
        "backup_model": MODELS_DIR / "utensil_detection" / "utensil_detector_yolo.pt",
        "architecture": "YOLO-v8",
        "input_size": 640,
        "classes": {
            0: {"name": "spoon", "length_cm": 18.0},
            1: {"name": "fork", "length_cm": 20.0}
        },
        "framework": "onnx",
        "size_mb": 12
    },

    # Depth estimation
    "depth_estimation": {
        "primary_model": MODELS_DIR / "depth_estimation" / "midas_small_depth.pth",
        "architecture": "MiDaS-Small",
        "framework": "pytorch",
        "size_mb": 82
    }
}

# Model loading priorities
MODEL_PRIORITY = {
    "segmentation": ["pytorch"],  # Only PyTorch working
    "classification": ["onnx"],   # ONNX preferred
    "utensil_detection": ["onnx", "pytorch"],  # ONNX primary
    "depth_estimation": ["pytorch"]  # PyTorch only
}

# Hardware requirements
HARDWARE_REQUIREMENTS = {
    "min_gpu_memory_gb": 6,
    "recommended_gpu": "RTX 3060 (12GB)",
    "min_ram_gb": 8,
    "cuda_version": "11.8+",
    "total_model_size_mb": 360
}

def get_model_path(model_type: str, priority: str = "primary") -> Path:
    """Get the path for a specific model."""
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Unknown model type: {model_type}")

    config = MODEL_CONFIG[model_type]

    if priority == "primary":
        return config["primary_model"]
    elif priority == "backup" and "backup_model" in config:
        return config["backup_model"]
    else:
        return config["primary_model"]

def validate_models() -> dict:
    """Validate that all required models exist."""
    status = {}

    for model_type, config in MODEL_CONFIG.items():
        primary_exists = config["primary_model"].exists()
        backup_exists = config.get("backup_model", Path("")).exists() if "backup_model" in config else True

        status[model_type] = {
            "primary_exists": primary_exists,
            "backup_exists": backup_exists,
            "size_mb": config["size_mb"],
            "framework": config["framework"],
            "working": primary_exists  # At minimum, primary must exist
        }

    return status

def get_total_model_size() -> float:
    """Calculate total size of all models in MB."""
    return sum(config["size_mb"] for config in MODEL_CONFIG.values())

if __name__ == "__main__":
    print("FOOD NUTRITION ANALYSIS - MODEL CONFIGURATION")
    print("=" * 50)

    # Validate models
    status = validate_models()

    for model_type, info in status.items():
        working_status = "✓" if info["working"] else "✗"
        print(f"{working_status} {model_type}: {info['size_mb']}MB ({info['framework']})")

        if not info["working"]:
            primary_path = get_model_path(model_type)
            print(f"   Missing: {primary_path}")

    print(f"\nTotal model size: {get_total_model_size():.0f}MB")

    # Check if all models are working
    all_working = all(info["working"] for info in status.values())
    print(f"All models ready: {'Yes' if all_working else 'No'}")