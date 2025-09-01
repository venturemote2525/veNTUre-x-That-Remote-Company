#!/usr/bin/env python3
"""
Volume Calculator

Enhanced 3D volume estimation from segmentation masks and depth maps.
Includes real-world scale calibration and volume confidence assessment.
"""

from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2
import logging
from scipy import ndimage


@dataclass
class VolumeResult:
    """Result of volume calculation for a food item."""
    
    volume_ml: float
    confidence: str  # "high", "medium", "low"
    mask_area_pixels: int
    avg_depth: float
    depth_variance: float
    surface_area_estimate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'volume_ml': round(self.volume_ml, 2),
            'confidence': self.confidence,
            'mask_area_pixels': self.mask_area_pixels,
            'avg_depth': round(self.avg_depth, 4),
            'depth_variance': round(self.depth_variance, 6),
            'surface_area_estimate': round(self.surface_area_estimate, 2)
        }


class VolumeCalculator:
    """
    Enhanced volume calculator using depth maps and segmentation masks.
    
    Provides multiple volume estimation methods and confidence assessment
    based on depth quality and mask characteristics.
    """
    
    def __init__(self, 
                 default_focal_length: float = 715.0,
                 default_camera_height_cm: float = 30.0,
                 pixel_to_mm_ratio: Optional[float] = None):
        """
        Initialize volume calculator.
        
        Args:
            default_focal_length: Camera focal length in pixels
            default_camera_height_cm: Typical camera height above food
            pixel_to_mm_ratio: Optional calibrated pixel-to-millimeter ratio
        """
        self.default_focal_length = default_focal_length
        self.default_camera_height_cm = default_camera_height_cm
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        
        # Confidence thresholds
        self.high_confidence_min_area = 1000  # pixels
        self.low_confidence_max_variance = 0.1
        self.high_confidence_max_variance = 0.05
        
        logging.info("Initialized VolumeCalculator with enhanced 3D estimation")
    
    def calculate_volume(self, mask: np.ndarray, depth_map: np.ndarray, 
                        method: str = "enhanced") -> VolumeResult:
        """
        Calculate volume of food item using mask and depth information.
        
        Args:
            mask: Binary segmentation mask (0-1 or 0-255)
            depth_map: Depth map from Depth Anything
            method: Volume calculation method ("simple", "enhanced", "surface_integration")
            
        Returns:
            VolumeResult with volume estimate and confidence
        """
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # Extract depth values within the mask
        masked_depth = depth_map[mask > 0]
        
        if len(masked_depth) == 0:
            return VolumeResult(0.0, "low", 0, 0.0, 0.0, 0.0)
        
        # Basic statistics
        mask_area = np.sum(mask > 0)
        avg_depth = np.mean(masked_depth)
        depth_variance = np.var(masked_depth)
        
        # Choose calculation method
        if method == "simple":
            volume_ml = self._calculate_simple_volume(mask_area, avg_depth)
        elif method == "enhanced":
            volume_ml = self._calculate_enhanced_volume(mask, depth_map)
        elif method == "surface_integration":
            volume_ml = self._calculate_surface_integration_volume(mask, depth_map)
        else:
            raise ValueError(f"Unknown volume calculation method: {method}")
        
        # Estimate surface area
        surface_area = self._estimate_surface_area(mask, depth_map)
        
        # Assess confidence
        confidence = self._assess_volume_confidence(
            mask_area, depth_variance, masked_depth, volume_ml
        )
        
        return VolumeResult(
            volume_ml=volume_ml,
            confidence=confidence,
            mask_area_pixels=int(mask_area),
            avg_depth=float(avg_depth),
            depth_variance=float(depth_variance),
            surface_area_estimate=float(surface_area)
        )
    
    def _calculate_simple_volume(self, mask_area: int, avg_depth: float) -> float:
        """Simple volume calculation: area × average depth × scale factor."""
        # Convert to real-world units (approximate)
        if self.pixel_to_mm_ratio:
            area_mm2 = mask_area * (self.pixel_to_mm_ratio ** 2)
            # Assume depth is relative, scale by typical food thickness
            depth_mm = avg_depth * 50.0  # Rough scaling factor
        else:
            # Use empirical scaling based on typical camera setup
            area_mm2 = mask_area * 0.25  # ~0.5mm per pixel squared
            depth_mm = avg_depth * 30.0   # Scale depth relatively
        
        volume_mm3 = area_mm2 * depth_mm
        volume_ml = volume_mm3 / 1000.0  # Convert mm³ to ml
        
        return max(0.0, volume_ml)
    
    def _calculate_enhanced_volume(self, mask: np.ndarray, depth_map: np.ndarray) -> float:
        """Enhanced volume calculation considering depth distribution."""
        # Extract masked depth region
        masked_depth = np.where(mask > 0, depth_map, 0)
        
        # Find depth layers/slices for better volume estimation
        depth_values = depth_map[mask > 0]
        if len(depth_values) == 0:
            return 0.0
        
        depth_min, depth_max = depth_values.min(), depth_values.max()
        
        # If depth range is very small, use simple calculation
        if (depth_max - depth_min) < 0.01:
            return self._calculate_simple_volume(np.sum(mask), np.mean(depth_values))
        
        # Layer-based integration
        num_layers = min(10, max(3, int((depth_max - depth_min) / 0.05)))
        depth_layers = np.linspace(depth_min, depth_max, num_layers + 1)
        
        total_volume = 0.0
        for i in range(len(depth_layers) - 1):
            layer_bottom = depth_layers[i]
            layer_top = depth_layers[i + 1]
            
            # Find pixels in this depth layer
            layer_mask = ((masked_depth >= layer_bottom) & 
                         (masked_depth < layer_top) & 
                         (mask > 0))
            
            layer_area = np.sum(layer_mask)
            if layer_area > 0:
                layer_thickness = layer_top - layer_bottom
                
                # Convert to real-world units
                if self.pixel_to_mm_ratio:
                    area_mm2 = layer_area * (self.pixel_to_mm_ratio ** 2)
                    thickness_mm = layer_thickness * 50.0
                else:
                    area_mm2 = layer_area * 0.25
                    thickness_mm = layer_thickness * 30.0
                
                layer_volume = area_mm2 * thickness_mm
                total_volume += layer_volume
        
        volume_ml = total_volume / 1000.0
        return max(0.0, volume_ml)
    
    def _calculate_surface_integration_volume(self, mask: np.ndarray, depth_map: np.ndarray) -> float:
        """Volume calculation using surface integration approach."""
        # Create 3D surface from depth map
        h, w = mask.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Extract masked coordinates and depths
        mask_indices = mask > 0
        if not np.any(mask_indices):
            return 0.0
        
        x_masked = x_coords[mask_indices]
        y_masked = y_coords[mask_indices]
        z_masked = depth_map[mask_indices]
        
        # Find the base plane (minimum depth in the mask region)
        base_depth = np.min(z_masked)
        
        # Calculate volume above the base plane
        height_above_base = z_masked - base_depth
        
        # Integrate using trapezoidal rule approximation
        if self.pixel_to_mm_ratio:
            pixel_area_mm2 = self.pixel_to_mm_ratio ** 2
            height_scale = 50.0
        else:
            pixel_area_mm2 = 0.25
            height_scale = 30.0
        
        # Sum all the volume elements
        volume_elements = height_above_base * height_scale * pixel_area_mm2
        total_volume_mm3 = np.sum(volume_elements)
        
        volume_ml = total_volume_mm3 / 1000.0
        return max(0.0, volume_ml)
    
    def _estimate_surface_area(self, mask: np.ndarray, depth_map: np.ndarray) -> float:
        """Estimate 3D surface area of the food item."""
        # Extract masked depth region
        masked_depth = np.where(mask > 0, depth_map, 0)
        
        # Calculate gradients to estimate surface slope
        grad_y, grad_x = np.gradient(masked_depth)
        
        # Calculate surface area elements considering slope
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        surface_factor = np.sqrt(1 + slope_magnitude**2)
        
        # Apply mask and sum surface elements
        surface_elements = surface_factor[mask > 0]
        
        if self.pixel_to_mm_ratio:
            pixel_area_mm2 = self.pixel_to_mm_ratio ** 2
        else:
            pixel_area_mm2 = 0.25
        
        total_surface_area_mm2 = np.sum(surface_elements) * pixel_area_mm2
        total_surface_area_cm2 = total_surface_area_mm2 / 100.0
        
        return max(0.0, total_surface_area_cm2)
    
    def _assess_volume_confidence(self, mask_area: int, depth_variance: float, 
                                 depth_values: np.ndarray, volume_ml: float) -> str:
        """Assess confidence in volume calculation based on various factors."""
        confidence_score = 0
        
        # Factor 1: Mask area (larger areas generally more reliable)
        if mask_area >= self.high_confidence_min_area:
            confidence_score += 2
        elif mask_area >= self.high_confidence_min_area // 2:
            confidence_score += 1
        
        # Factor 2: Depth variance (lower variance = more consistent depth)
        if depth_variance <= self.high_confidence_max_variance:
            confidence_score += 2
        elif depth_variance <= self.low_confidence_max_variance:
            confidence_score += 1
        
        # Factor 3: Depth distribution
        if len(depth_values) > 100:  # Sufficient depth samples
            depth_range = depth_values.max() - depth_values.min()
            if 0.05 < depth_range < 0.5:  # Reasonable depth variation
                confidence_score += 1
        
        # Factor 4: Volume reasonableness (10ml to 2L for food portions)
        if 10.0 <= volume_ml <= 2000.0:
            confidence_score += 1
        
        # Determine confidence level
        if confidence_score >= 5:
            return "high"
        elif confidence_score >= 3:
            return "medium"
        else:
            return "low"
    
    def calibrate_with_reference_object(self, reference_mask: np.ndarray, 
                                       reference_volume_ml: float,
                                       reference_depth: np.ndarray) -> float:
        """
        Calibrate volume calculation using a reference object of known volume.
        
        Args:
            reference_mask: Mask of reference object
            reference_volume_ml: Known volume of reference object
            reference_depth: Depth map containing reference object
            
        Returns:
            Calibration scale factor to improve volume estimates
        """
        # Calculate volume using current method
        calculated_volume = self._calculate_enhanced_volume(reference_mask, reference_depth)
        
        if calculated_volume > 0:
            scale_factor = reference_volume_ml / calculated_volume
            logging.info(f"Calibration scale factor: {scale_factor:.3f}")
            return scale_factor
        else:
            logging.warning("Could not calculate reference object volume")
            return 1.0
    
    def set_camera_calibration(self, focal_length: float, camera_height_cm: float,
                              pixel_to_mm_ratio: float):
        """Set camera calibration parameters for more accurate volume estimation."""
        self.default_focal_length = focal_length
        self.default_camera_height_cm = camera_height_cm
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        
        logging.info(f"Updated camera calibration: focal={focal_length}, "
                    f"height={camera_height_cm}cm, pixel_ratio={pixel_to_mm_ratio}")
    
    def estimate_food_height(self, mask: np.ndarray, depth_map: np.ndarray) -> float:
        """Estimate the height of food item in real-world units (cm)."""
        depth_values = depth_map[mask > 0]
        if len(depth_values) == 0:
            return 0.0
        
        depth_range = depth_values.max() - depth_values.min()
        
        # Convert relative depth to real height estimate
        # This is approximate and depends on camera setup
        height_cm = depth_range * 15.0  # Empirical scaling factor
        
        return max(0.0, min(height_cm, 30.0))  # Cap at reasonable food heights