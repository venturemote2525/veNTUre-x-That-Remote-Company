#!/usr/bin/env python3
"""
Run Nutrition Analysis with Reference-Scale Calibration (CLI)

Example:
  python src/pipeline/cli/reference_scale_run.py --image path/to/food.jpg \
      --out outputs/food.json --confidence 0.5

Inside Docker container:
  docker exec -it nutrition-analysis-app \
    python src/pipeline/cli/reference_scale_run.py --image /app/data/food.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict

import cv2

from src.pipeline.nutrition_pipeline import NutritionPipeline
from src.reference.reference_scale import ReferenceObjectCatalog


def load_reference_overrides(path: Optional[str]) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference objects file not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PyYAML is required to load {path}: {e}")

    with open(p, "r") as f:
        data: Dict[str, float] = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("reference objects YAML must be a mapping of name: length_mm")

    # Override in-memory catalog
    ReferenceObjectCatalog.DEFAULT_SIZES_MM.update({k: float(v) for k, v in data.items()})


def main():
    ap = argparse.ArgumentParser(description="Nutrition analysis with reference-scale calibration")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--out", default=None, help="Optional path to write JSON output")
    ap.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    ap.add_argument("--depth-encoder", default="vitb", help="Depth Anything encoder (vits|vitb|vitl)")
    ap.add_argument("--reference-objects", default=str(Path(__file__).with_name("reference_objects.yaml")),
                    help="YAML file with reference object sizes (mm)")

    args = ap.parse_args()

    # Optional overrides for reference object catalog
    load_reference_overrides(args.reference_objects)

    # Create pipeline (reference detector is enabled by default in constructor)
    pipeline = NutritionPipeline(
        depth_encoder=args.depth_encoder,
        specific_classifier=None,
        reference_detector=None,  # use heuristic default
        enable_reference_scale=True,
    )

    # Analyze
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = pipeline.analyze_food_image(str(image_path), conf_threshold=args.confidence)

    # Emit JSON
    data = json.loads(result.export_json())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
