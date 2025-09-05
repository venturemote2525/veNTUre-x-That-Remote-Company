#!/usr/bin/env python3
"""
Learned Utensil Detector

Uses a trained segmentation model to detect spoons and forks for accurate scale detection.
Replaces the heuristic detector with deep learning-based detection.
"""

from typing import List, Optional, Tuple, Dict
import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .reference_scale import ReferenceObjectDetectorInterface, ReferenceObjectMeasurement
from .utensil_measurements import UtensilMeasurements, get_utensil_length_mm


class UtensilUNet(nn.Module):
    """UNet model for utensil segmentation (same as training script)."""
    
    def __init__(self, num_classes=3, backbone='resnet34'):
        super(UtensilUNet, self).__init__()
        
        # Encoder (ResNet backbone)
        if backbone == 'resnet34':
            resnet = models.resnet34(weights='DEFAULT')
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT')
        else:
            resnet = models.resnet34(weights='DEFAULT')
            
        # Remove last two layers (avgpool and fc)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        output = self.decoder(features)
        
        # Resize to input size if needed
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return output


class LearnedUtensilDetector(ReferenceObjectDetectorInterface):
    """
    Learned utensil detector using trained segmentation model.
    
    Detects spoons and forks in images and calculates accurate scale measurements
    based on their real-world dimensions.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 confidence_threshold: float = 0.7,
                 min_area_pixels: int = 500,
                 device: Optional[str] = None):
        """
        Initialize the learned utensil detector.
        
        Args:
            model_path: Path to trained utensil segmentation model
            confidence_threshold: Minimum confidence for detections
            min_area_pixels: Minimum area in pixels for valid detections
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_area_pixels = min_area_pixels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.utensil_measurements = UtensilMeasurements()
        self.class_names = ['background', 'spoon', 'fork']
        
        # Model and transform (loaded lazily)
        self._model = None
        self._transform = None
        
        # Try to find model if not provided
        if not self.model_path:
            self.model_path = self._find_trained_model()
    
    def _find_trained_model(self) -> Optional[str]:
        """Try to find a trained utensil model."""
        possible_paths = [
            Path(__file__).parent.parent / "segmentation" / "utensils" / "training" / "checkpoints" / "utensil_unet_best.pth",
            Path(__file__).parent.parent / "segmentation" / "utensils" / "training" / "checkpoints" / "utensil_unet_final.pth",
        ]
        
        for path in possible_paths:
            if path.exists():
                logging.info(f"Found utensil model: {path}")
                return str(path)
        
        logging.warning("No trained utensil model found. Use heuristic detector as fallback.")
        return None
    
    def _load_model(self):
        """Load the trained utensil segmentation model."""
        if self._model is not None:
            return
        
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"Utensil model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model config
        config = checkpoint.get('config', {})
        num_classes = config.get('num_classes', 3)
        backbone = config.get('backbone', 'resnet34')
        
        # Create model
        self._model = UtensilUNet(num_classes=num_classes, backbone=backbone)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        self._model.to(self.device)
        
        # Create transform
        self._transform = A.Compose([
            A.Resize(512, 512),  # Match training size
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        logging.info(f"Loaded utensil model from {self.model_path}")
    
    def detect(self, image: np.ndarray) -> List[ReferenceObjectMeasurement]:
        """
        Detect utensils in image and calculate scale measurements.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected utensil measurements
        """
        if image is None or image.size == 0:
            return []
        
        try:
            self._load_model()
        except Exception as e:
            logging.warning(f"Failed to load utensil model: {e}")
            return []
        
        detections = []
        
        try:
            # Preprocess image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_shape = rgb_image.shape[:2]
            
            # Apply transform
            transformed = self._transform(image=rgb_image)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(input_tensor)
                pred_mask = torch.argmax(outputs, dim=1)[0]  # Remove batch dimension
                pred_probs = torch.softmax(outputs, dim=1)[0]  # Get probabilities
            
            # Resize predictions back to original size
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0).float(),
                size=original_shape,
                mode='nearest'
            )[0, 0].cpu().numpy().astype(np.uint8)
            
            pred_probs = F.interpolate(
                pred_probs.unsqueeze(0),
                size=original_shape,
                mode='bilinear',
                align_corners=False
            )[0].cpu().numpy()
            
            # Process each class (skip background)
            for class_idx in range(1, len(self.class_names)):
                class_name = self.class_names[class_idx]
                
                # Create binary mask for this class
                class_mask = (pred_mask == class_idx).astype(np.uint8)
                
                if class_mask.sum() < self.min_area_pixels:
                    continue
                
                # Get confidence from probability map
                confidence = float(pred_probs[class_idx][class_mask > 0].mean())
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Find connected components
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_area_pixels:
                        continue
                    
                    # Get bounding box and measurements
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = (x, y, x + w, y + h)
                    
                    # Calculate utensil length (use longest dimension of bounding box)
                    pixel_length = float(max(w, h))
                    
                    # Determine utensil type and get real-world size
                    utensil_type = self._classify_utensil_type(class_name, w, h, area)
                    real_length_mm = self.utensil_measurements.get_length_mm(utensil_type)
                    
                    if real_length_mm and pixel_length > 10:
                        scale_mm_per_pixel = real_length_mm / pixel_length
                        
                        # Create measurement
                        measurement = ReferenceObjectMeasurement(
                            object_type=utensil_type,
                            bbox=bbox,
                            mask=class_mask,
                            pixel_length=pixel_length,
                            known_real_length_mm=real_length_mm,
                            scale_mm_per_pixel=scale_mm_per_pixel,
                            confidence=confidence
                        )
                        
                        detections.append(measurement)
            
            # Sort by confidence
            detections.sort(key=lambda d: d.confidence, reverse=True)
            
            if detections:
                logging.info(f"Detected {len(detections)} utensils with confidences: {[d.confidence for d in detections]}")
            
        except Exception as e:
            logging.error(f"Utensil detection failed: {e}")
        
        return detections
    
    def _classify_utensil_type(self, base_class: str, width: int, height: int, area: int) -> str:
        """
        Classify specific utensil type based on shape analysis.
        
        Args:
            base_class: Base class ('spoon' or 'fork')
            width: Bounding box width
            height: Bounding box height  
            area: Contour area
            
        Returns:
            Specific utensil type (e.g., 'dinner_spoon', 'dessert_fork')
        """
        # Calculate shape metrics
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        bbox_area = width * height
        fill_ratio = area / (bbox_area + 1e-6)
        
        # Size-based classification (relative to typical image sizes)
        max_dimension = max(width, height)
        
        if base_class == 'spoon':
            # Classify spoon size
            if max_dimension > 150:
                return 'dinner_spoon'
            elif max_dimension > 100:
                return 'dessert_spoon'
            else:
                return 'teaspoon'
        
        elif base_class == 'fork':
            # Classify fork size
            if max_dimension > 150:
                return 'dinner_fork'
            else:
                return 'dessert_fork'
        
        # Fallback to generic type
        return base_class
    
    def get_supported_objects(self) -> List[str]:
        """Get list of supported utensil types."""
        return self.utensil_measurements.get_supported_utensils()
    
    def is_available(self) -> bool:
        """Check if the detector is available (model exists)."""
        return self.model_path is not None and Path(self.model_path).exists()


class HybridUtensilDetector(ReferenceObjectDetectorInterface):
    """
    Hybrid detector that tries learned detection first, falls back to heuristic.
    
    Provides robust utensil detection by combining deep learning and traditional CV.
    """
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """Initialize hybrid detector."""
        # Try learned detector first
        self.learned_detector = LearnedUtensilDetector(model_path, **kwargs)
        
        # Import heuristic detector as fallback
        from .reference_scale import HeuristicReferenceObjectDetector
        self.heuristic_detector = HeuristicReferenceObjectDetector(['spoon', 'fork'])
        
        self.use_learned = self.learned_detector.is_available()
        
        if self.use_learned:
            logging.info("Using learned utensil detector")
        else:
            logging.info("Using heuristic utensil detector (no trained model found)")
    
    def detect(self, image: np.ndarray) -> List[ReferenceObjectMeasurement]:
        """Detect utensils using best available method."""
        if self.use_learned:
            detections = self.learned_detector.detect(image)
            if detections:
                return detections
        
        # Fallback to heuristic detection
        return self.heuristic_detector.detect(image)
    
    def get_supported_objects(self) -> List[str]:
        """Get supported objects from active detector."""
        if self.use_learned:
            return self.learned_detector.get_supported_objects()
        return self.heuristic_detector.get_supported_objects()