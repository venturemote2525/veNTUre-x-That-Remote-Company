#!/usr/bin/env python3
"""
Main Nutrition Analysis Pipeline

Integrates food segmentation (Mask R-CNN), depth estimation (Depth Anything),
and nutritional analysis for comprehensive food portion analysis.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from datetime import datetime

import numpy as np
import torch
import cv2
from PIL import Image

# Add model paths to system path
REPO_ROOT = Path(__file__).resolve().parents[2]
# Ensure repo src on path
sys.path.append(str(REPO_ROOT / 'src'))

DEPTH_PATH = REPO_ROOT / 'src' / 'depth' / 'depth_anything'
MASK_RCNN_PATH = REPO_ROOT / 'src' / 'segmentation' / 'mask_rcnn'

sys.path.append(str(DEPTH_PATH))
sys.path.append(str(MASK_RCNN_PATH))
sys.path.append(str(MASK_RCNN_PATH / "utils"))

try:
    from depth_anything.dpt import DepthAnything
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    import torch.nn.functional as F
except ImportError as e:
    logging.warning(f"Could not import depth_anything modules: {e}")

try:
    from train_maskrcnn_food import FoodMaskRCNNTrainer
    from utils.config import FoodMaskRCNNConfig
except ImportError as e:
    logging.warning(f"Could not import mask_rcnn modules: {e}")

from src.nutrition.database import NutritionDatabase, NutritionInfo
from src.volume.volume_calculator import VolumeCalculator, VolumeResult
from src.specific_classification.interface import SpecificClassifierInterface
from src.reference.reference_scale import (
    ReferenceObjectDetectorInterface,
    HeuristicReferenceObjectDetector,
    ReferenceObjectMeasurement,
    choose_best_scale,
)


class FoodItem:
    """Represents a detected food item with segmentation and classification info."""
    
    def __init__(self, item_id: int, broad_class: str, mask: np.ndarray, 
                 bbox: Tuple[int, int, int, int], confidence_broad: float):
        self.item_id = item_id
        self.broad_class = broad_class
        self.specific_class: Optional[str] = None
        self.confidence_broad = confidence_broad
        self.confidence_specific: Optional[float] = None
        self.mask = mask
        self.bbox = bbox  # (x1, y1, x2, y2)


class FoodItemAnalysis:
    """Complete analysis result for a single food item."""
    
    def __init__(self, food_item: FoodItem, volume_result: VolumeResult, 
                 nutrition_info: NutritionInfo):
        self.food_item = food_item
        self.volume = volume_result
        self.nutrition = nutrition_info
        
        # Calculate total nutrition based on estimated volume/weight
        weight_g = volume_result.volume_ml * nutrition_info.density_g_ml
        self.total_calories = (weight_g / 100.0) * nutrition_info.calories_per_100g
        self.total_protein_g = (weight_g / 100.0) * nutrition_info.protein_g
        self.total_carbs_g = (weight_g / 100.0) * nutrition_info.carbs_g
        self.total_fat_g = (weight_g / 100.0) * nutrition_info.fat_g
        self.total_fiber_g = (weight_g / 100.0) * nutrition_info.fiber_g
        self.estimated_weight_g = weight_g
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'item_id': self.food_item.item_id,
            'broad_class': self.food_item.broad_class,
            'specific_class': self.food_item.specific_class,
            'confidence_broad': self.food_item.confidence_broad,
            'confidence_specific': self.food_item.confidence_specific,
            'volume_ml': self.volume.volume_ml,
            'volume_confidence': self.volume.confidence,
            'estimated_weight_g': self.estimated_weight_g,
            'nutrition': {
                'calories': self.total_calories,
                'protein_g': self.total_protein_g,
                'carbs_g': self.total_carbs_g,
                'fat_g': self.total_fat_g,
                'fiber_g': self.total_fiber_g
            }
        }


class NutritionAnalysisResult:
    """Complete nutrition analysis result for an image."""
    
    def __init__(self, image_path: str, food_items: List[FoodItemAnalysis]):
        self.image_path = image_path
        self.food_items = food_items
        self.analysis_timestamp = datetime.now().isoformat()
        
        # Calculate totals
        self.total_calories = sum(item.total_calories for item in food_items)
        self.total_volume_ml = sum(item.volume.volume_ml for item in food_items)
        self.total_protein_g = sum(item.total_protein_g for item in food_items)
        self.total_carbs_g = sum(item.total_carbs_g for item in food_items)
        self.total_fat_g = sum(item.total_fat_g for item in food_items)
        self.total_fiber_g = sum(item.total_fiber_g for item in food_items)
        self.total_weight_g = sum(item.estimated_weight_g for item in food_items)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'image_path': self.image_path,
            'analysis_timestamp': self.analysis_timestamp,
            'summary': {
                'total_calories': round(self.total_calories, 1),
                'total_volume_ml': round(self.total_volume_ml, 1),
                'total_weight_g': round(self.total_weight_g, 1),
                'total_protein_g': round(self.total_protein_g, 1),
                'total_carbs_g': round(self.total_carbs_g, 1),
                'total_fat_g': round(self.total_fat_g, 1),
                'total_fiber_g': round(self.total_fiber_g, 1)
            },
            'food_items': [item.to_dict() for item in self.food_items],
            'food_count': len(self.food_items)
        }
    
    def export_json(self, output_path: Optional[str] = None) -> str:
        """Export results to JSON format."""
        result_dict = self.to_dict()
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
        return json.dumps(result_dict, indent=2)


class NutritionPipeline:
    """Main pipeline for integrated food nutrition analysis."""
    
    def __init__(self, 
                 segmentation_checkpoint: Optional[str] = None,
                 depth_encoder: str = "vitb",
                 device: Optional[str] = None,
                 specific_classifier: Optional[SpecificClassifierInterface] = None,
                 reference_detector: Optional[ReferenceObjectDetectorInterface] = None,
                 enable_reference_scale: bool = True):
        """
        Initialize the nutrition analysis pipeline.
        
        Args:
            segmentation_checkpoint: Path to Mask R-CNN checkpoint
            depth_encoder: Depth Anything encoder ('vits', 'vitb', 'vitl')
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            specific_classifier: Optional specific food classifier from your friend
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_checkpoint = segmentation_checkpoint
        self.depth_encoder = depth_encoder
        
        # Initialize components
        self.nutrition_db = NutritionDatabase()
        self.volume_calculator = VolumeCalculator()
        self.specific_classifier = specific_classifier
        self.reference_detector = reference_detector or HeuristicReferenceObjectDetector()
        self.enable_reference_scale = enable_reference_scale
        
        # Model placeholders (loaded lazily)
        self._segmentation_model = None
        self._depth_model = None
        self._depth_transform = None
        
        # Swiss 7-class food categories
        self.class_names = ["fruit", "vegetable", "carbohydrate", "protein", "dairy", "fat", "other"]
        
        logging.info(f"Initialized NutritionPipeline on {self.device}")
    
    def _load_segmentation_model(self):
        """Load Mask R-CNN segmentation model."""
        if self._segmentation_model is not None:
            return
            
        if not self.segmentation_checkpoint:
            # Try to find latest checkpoint
            checkpoint_dirs = self._find_checkpoint_directories()
            if checkpoint_dirs:
                self.segmentation_checkpoint = self._find_best_checkpoint(checkpoint_dirs[0])
            else:
                raise ValueError("No segmentation checkpoint provided and none found automatically")
        
        checkpoint = torch.load(self.segmentation_checkpoint, map_location=self.device)
        cfg_dict = checkpoint.get("config", {})
        
        cfg = FoodMaskRCNNConfig(
            name=cfg_dict.get("name", "nutrition_analysis"),
            class_names=self.class_names,
            backbone=cfg_dict.get("backbone", "resnet50"),
            epochs=1,
            batch_size=1,
            image_min_size=cfg_dict.get("image_min_size", 640),
            image_max_size=cfg_dict.get("image_max_size", 800),
            checkpoint_dir=str(Path(self.segmentation_checkpoint).parent),
            log_dir=str(Path(self.segmentation_checkpoint).parent.parent / "logs")
        )
        
        trainer = FoodMaskRCNNTrainer(cfg)
        trainer.build_model()
        state = checkpoint.get("model_state_dict") or checkpoint
        trainer.model.load_state_dict(state)
        trainer.model.eval().to(self.device)
        
        self._segmentation_model = trainer.model
        logging.info(f"Loaded segmentation model from {self.segmentation_checkpoint}")
    
    def _load_depth_model(self):
        """Load Depth Anything model."""
        if self._depth_model is not None:
            return
            
        model_id = f"LiheYoung/depth_anything_{self.depth_encoder}14"
        self._depth_model = DepthAnything.from_pretrained(model_id).to(self.device).eval()
        
        self._depth_transform = Compose([
            Resize(
                width=518, height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        logging.info(f"Loaded depth model with encoder {self.depth_encoder}")
    
    def _find_checkpoint_directories(self) -> List[Path]:
        """Find available checkpoint directories."""
        candidates = [
            REPO_ROOT / "src" / "training" / "checkpoints",
            REPO_ROOT / "checkpoints",
        ]
        found = []
        for base in candidates:
            if base.exists():
                for sub in base.iterdir():
                    if sub.is_dir() and sub.name.startswith("swiss_7class_"):
                        found.append(sub)
        return sorted(found)
    
    def _find_best_checkpoint(self, ckpt_dir: Path) -> Optional[Path]:
        """Find the best checkpoint in a directory."""
        if (ckpt_dir / "best_checkpoint.pth").exists():
            return ckpt_dir / "best_checkpoint.pth"
        if (ckpt_dir / "latest_checkpoint.pth").exists():
            return ckpt_dir / "latest_checkpoint.pth"
        epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        return epoch_ckpts[-1] if epoch_ckpts else None
    
    def _segment_image(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[FoodItem]:
        """Run segmentation on image and return food items."""
        self._load_segmentation_model()
        
        # Convert BGR to tensor
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self._segmentation_model(tensor)
        
        pred = predictions[0]
        boxes = pred.get("boxes", torch.empty(0, 4))
        scores = pred.get("scores", torch.empty(0))
        labels = pred.get("labels", torch.empty(0))
        masks = pred.get("masks", torch.empty(0, 0, 0))
        
        if boxes.numel() == 0:
            return []
        
        # Convert to numpy
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        # Filter by confidence
        keep = scores >= conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        food_items = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if len(masks) > i:
                mask = masks[i].detach().cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
            else:
                # Create mask from bounding box if no mask available
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = box.astype(int)
                mask[y1:y2, x1:x2] = 1
            
            # Map label to class name (labels are 1-indexed in Mask R-CNN)
            class_idx = int(label) - 1
            if 0 <= class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
            else:
                class_name = "other"
            
            bbox = tuple(box.astype(int))
            food_item = FoodItem(i, class_name, mask, bbox, float(score))
            food_items.append(food_item)
        
        return food_items
    
    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map for image."""
        self._load_depth_model()
        
        h, w = image.shape[:2]
        
        # Preprocess
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        net_input = self._depth_transform({"image": rgb})["image"]
        net_input = torch.from_numpy(net_input).unsqueeze(0).to(self.device)
        
        # Estimate depth
        with torch.no_grad():
            depth = self._depth_model(net_input)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = depth.cpu().numpy()
        
        return depth

    def _estimate_scale_mm_per_pixel(self, image: np.ndarray) -> Optional[float]:
        """Estimate mm-per-pixel scale from reference object(s) if enabled."""
        if not self.enable_reference_scale or self.reference_detector is None:
            return None
        try:
            detections = self.reference_detector.detect(image)
            scale = choose_best_scale(detections)
            if scale:
                logging.info(f"Estimated scale: {scale:.4f} mm/px using reference object")
            return scale
        except Exception as e:
            logging.warning(f"Reference scale estimation failed: {e}")
            return None
    
    def _apply_specific_classification(self, image: np.ndarray, food_items: List[FoodItem]):
        """Apply specific classification if available."""
        if self.specific_classifier is None or not self.specific_classifier.is_available():
            return
        
        for food_item in food_items:
            try:
                x1, y1, x2, y2 = food_item.bbox
                roi = image[y1:y2, x1:x2]
                
                specific_result = self.specific_classifier.classify(roi, food_item.bbox)
                food_item.specific_class = specific_result.specific_class
                food_item.confidence_specific = specific_result.confidence
            except Exception as e:
                logging.warning(f"Specific classification failed for item {food_item.item_id}: {e}")
    
    def analyze_food_image(self, image_path: str, conf_threshold: float = 0.5) -> NutritionAnalysisResult:
        """
        Analyze a food image for nutrition content.
        
        Args:
            image_path: Path to food image
            conf_threshold: Minimum confidence for detections
            
        Returns:
            Complete nutrition analysis result
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logging.info(f"Analyzing image: {image_path}")
        
        # Step 1: Segment food items
        food_items = self._segment_image(image, conf_threshold)
        logging.info(f"Detected {len(food_items)} food items")
        
        if not food_items:
            return NutritionAnalysisResult(image_path, [])
        
        # Step 2: Estimate reference scale (optional)
        scale_mm_per_px = self._estimate_scale_mm_per_pixel(image)
        if scale_mm_per_px:
            # Update calculator calibration
            self.volume_calculator.set_camera_calibration(
                focal_length=self.volume_calculator.default_focal_length,
                camera_height_cm=self.volume_calculator.default_camera_height_cm,
                pixel_to_mm_ratio=scale_mm_per_px,
            )
        
        # Step 3: Estimate depth
        depth_map = self._estimate_depth(image)
        logging.info("Completed depth estimation")
        
        # Step 4: Apply specific classification if available
        self._apply_specific_classification(image, food_items)
        
        # Step 5: Analyze each food item
        analyses = []
        for food_item in food_items:
            # Calculate volume
            volume_result = self.volume_calculator.calculate_volume(
                food_item.mask, depth_map
            )
            
            # Get nutrition info (prefer specific over broad classification)
            nutrition_key = food_item.specific_class or food_item.broad_class
            nutrition_info = self.nutrition_db.get_nutrition_info(nutrition_key)
            
            # Create analysis
            analysis = FoodItemAnalysis(food_item, volume_result, nutrition_info)
            analyses.append(analysis)
        
        result = NutritionAnalysisResult(image_path, analyses)
        logging.info(f"Analysis complete: {result.total_calories:.1f} calories, {len(analyses)} items")

        return result
    
    def process_batch(self, image_paths: List[str], output_dir: Optional[str] = None) -> List[NutritionAnalysisResult]:
        """Process multiple images and optionally save results."""
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logging.info(f"Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                result = self.analyze_food_image(image_path)
                results.append(result)
                
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(exist_ok=True, parents=True)
                    
                    # Save individual result
                    result_file = output_path / f"{Path(image_path).stem}_nutrition.json"
                    result.export_json(str(result_file))
                    
            except Exception as e:
                logging.error(f"Failed to process {image_path}: {e}")
        
        # Save batch summary
        if output_dir and results:
            batch_summary = {
                'batch_summary': {
                    'total_images': len(image_paths),
                    'successful_analyses': len(results),
                    'total_calories': sum(r.total_calories for r in results),
                    'total_items': sum(len(r.food_items) for r in results)
                },
                'results': [r.to_dict() for r in results]
            }
            
            batch_file = Path(output_dir) / "batch_nutrition_analysis.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            
            logging.info(f"Batch results saved to {batch_file}")
        
        return results
