#!/usr/bin/env python3
"""
ONNX-based Segmentation Wrapper for Food Analysis API
"""
import numpy as np
from pathlib import Path
import cv2
import onnxruntime as ort
from typing import Dict, List

# Configuration
CONFIG = {
    'num_classes': 8,
    'classes': ['background', 'fruit', 'vegetable', 'carbohydrate', 'protein', 'dairy', 'fat', 'other'],
    'image_size': 512,
}

class ONNXSegmentationModel:
    """ONNX-based segmentation model for food analysis"""

    def __init__(self):
        self.session = None
        self.class_names = CONFIG['classes']
        self.image_size = CONFIG['image_size']
        self.load_model()

    def load_model(self):
        """Load ONNX segmentation model"""
        model_path = Path(__file__).parent / "models" / "segmentation" / "segformer_food_segmentation.onnx"

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(str(model_path))

        print(f"[OK] ONNX Segmentation model loaded from: {model_path}")

    def preprocess(self, image_array):
        """
        Preprocess image for segmentation model

        Args:
            image_array: numpy array in RGB format (H, W, 3)

        Returns:
            preprocessed tensor ready for inference
        """
        # Resize to model input size
        resized = cv2.resize(image_array, (self.image_size, self.image_size))

        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        normalized = (resized.astype(np.float32) / 255.0 - mean) / std

        # Convert to CHW format and add batch dimension
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        return tensor

    def predict(self, image_array, confidence_threshold=0.5):
        """
        Run segmentation on image

        Args:
            image_array: numpy array in RGB format (H, W, 3)
            confidence_threshold: minimum confidence for detections

        Returns:
            dict with 'detections', 'detections_count', 'mask_shape'
        """
        original_h, original_w = image_array.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(image_array)

        # Run ONNX inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        # Get logits (output shape: [1, num_classes, H, W])
        logits = outputs[0]

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get predicted classes for visualization only
        pred_classes = np.argmax(logits, axis=1).squeeze(0)

        # Get probability maps for each class (model resolution)
        prob_maps = probs.squeeze(0)

        # Resize predicted classes to original size (for optional visualization)
        pred_mask_resized = cv2.resize(
            pred_classes.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )

        # Instance extraction per class using connected components on probability threshold
        detections: List[Dict] = []
        min_area = int(original_w * original_h * 0.005)  # ~0.5% of image area

        for class_id in range(1, len(self.class_names)):  # Skip background
            # Resize probability map to original size
            prob_map_class = cv2.resize(
                prob_maps[class_id], (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )
            # Threshold on probability
            prob_mask = (prob_map_class >= confidence_threshold).astype(np.uint8)
            total_pixels = int(prob_mask.sum())
            if total_pixels == 0:
                continue

            # Connected components on thresholded mask
            num_labels, labels = cv2.connectedComponents(prob_mask)
            for lbl in range(1, num_labels):
                comp_mask = (labels == lbl)
                area = int(comp_mask.sum())
                if area < min_area:
                    continue
                # Mean confidence over component
                confidence = float(prob_map_class[comp_mask].mean())
                # BBox in original coords
                ys, xs = np.where(comp_mask)
                if ys.size == 0:
                    continue
                y_min, y_max = int(ys.min()), int(ys.max())
                x_min, x_max = int(xs.min()), int(xs.max())

                detections.append({
                    "class_id": int(class_id),
                    "class_name": self.class_names[class_id],
                    "confidence": confidence,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area_pixels": area,
                    "mask": comp_mask,  # per-instance mask at original resolution
                    "specific_food": None,
                })

        return {
            "detections": detections,
            "detections_count": len(detections),
            "mask_shape": pred_mask_resized.shape
        }

# Singleton instance
_model_instance = None

def get_segmentation_model():
    """Get or create singleton model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = ONNXSegmentationModel()
    return _model_instance
