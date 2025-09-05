#!/usr/bin/env python3
"""
Advanced Bring+Swiss 7-Class Food Segmentation App

Tests the advanced SegFormer model trained on combined Bring+Swiss dataset.
Shows segmentation results with 7 food categories using transformer architecture.

Classes: fruit, vegetable, carbohydrate, protein, dairy, fat, other
"""

import sys
import os
from pathlib import Path
import tempfile
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:
    # Try alternative import paths
    try:
        from transformers.models.segformer import SegformerForSemanticSegmentation
    except ImportError:
        # Use AutoModel as fallback
        from transformers import AutoModel
        SegformerForSemanticSegmentation = AutoModel
from scipy.ndimage import label as connected_components
from skimage.measure import regionprops
import math
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

# Import reference detection (with error handling)
try:
    from reference.utensil_detector import HybridUtensilDetector, LearnedUtensilDetector
    from reference.reference_scale import ReferenceObjectMeasurement
    REFERENCE_DETECTION_AVAILABLE = True
except ImportError as e:
    st.warning(f"Reference utensil detection not available: {e}")
    REFERENCE_DETECTION_AVAILABLE = False

# YOLO Utensil Detector Class
class YOLOUtensilDetector:
    """Simple YOLO-based utensil detector for reference scaling."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = ['spoon', 'fork']  # Based on your YOLO training
    
    def detect(self, image):
        """Detect utensils using YOLO model."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except ImportError:
                st.error("ultralytics package not available for YOLO detection")
                return []
            except Exception as e:
                st.error(f"Failed to load YOLO model: {e}")
                return []
        
        detections = []
        try:
            # Run YOLO inference
            results = self.model(image, conf=0.5)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Get class name
                        if cls_id < len(self.class_names):
                            object_type = self.class_names[cls_id]
                        else:
                            object_type = f"utensil_{cls_id}"
                        
                        # Calculate scale (rough approximation)
                        # Assume average spoon length = 150mm, fork = 180mm
                        known_lengths = {'spoon': 150, 'fork': 180}
                        known_length_mm = known_lengths.get(object_type, 150)
                        
                        # Calculate pixel length (diagonal of bounding box)
                        pixel_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        scale_mm_per_pixel = known_length_mm / pixel_length if pixel_length > 0 else 0
                        
                        # Create detection object
                        detection = ReferenceObjectMeasurement(
                            object_type=object_type,
                            bbox=(x1, y1, x2, y2),
                            mask=None,
                            pixel_length=pixel_length,
                            known_real_length_mm=known_length_mm,
                            scale_mm_per_pixel=scale_mm_per_pixel,
                            confidence=conf
                        )
                        detections.append(detection)
            
        except Exception as e:
            st.error(f"YOLO detection failed: {e}")
            return []
        
        return detections

def generate_depth_map(image_rgb):
    """Generate depth map using simple and fast estimation."""
    try:
        # Use fast gradient-based depth estimation (no model download needed)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale depth estimation
        # 1. Edge-based depth (edges are usually closer)
        edges = cv2.Canny(gray, 50, 150)
        edge_depth = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        # 2. Gradient magnitude (texture indicates depth)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3. Brightness-based depth (darker areas often further)
        brightness_depth = 255 - gray
        
        # 4. Laplacian (focus measure - focused areas closer)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_depth = np.abs(laplacian)
        
        # Combine depth cues
        combined_depth = (
            0.3 * edge_depth +
            0.2 * cv2.GaussianBlur(grad_mag.astype(np.float32), (15, 15), 0) +
            0.2 * cv2.GaussianBlur(brightness_depth.astype(np.float32), (15, 15), 0) +
            0.3 * cv2.GaussianBlur(laplacian_depth.astype(np.float32), (15, 15), 0)
        )
        
        # Smooth the result
        depth_smooth = cv2.GaussianBlur(combined_depth, (21, 21), 0)
        
        # Normalize to 0-255
        depth_normalized = ((depth_smooth - depth_smooth.min()) / (depth_smooth.max() - depth_smooth.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap (plasma: purple=close, yellow=far)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        return depth_colored
        
    except Exception as e:
        st.error(f"Depth estimation failed: {e}")
        # Return a simple gradient as fallback
        h, w = image_rgb.shape[:2]
        gradient = np.linspace(0, 255, h).reshape(-1, 1).repeat(w, axis=1).astype(np.uint8)
        gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_PLASMA)
        gradient_colored = cv2.cvtColor(gradient_colored, cv2.COLOR_BGR2RGB)
        return gradient_colored

def create_depth_segmentation_visualization(image_rgb, depth_map, detections, volume_results=None):
    """Create visualization combining depth map with segmentation and volume labels."""
    try:
        if depth_map is None:
            return None
        
        # Create base visualization
        vis_image = depth_map.copy()
        
        # Overlay segmentation masks with transparency
        if detections:
            # Create segmentation overlay
            seg_overlay = np.zeros_like(image_rgb)
            
            # Color mapping for food classes
            colors = {
                'fruit': (255, 100, 100),     # Light red
                'vegetable': (100, 255, 100), # Light green
                'carbohydrate': (100, 100, 255), # Light blue
                'protein': (255, 100, 255),   # Light magenta
                'dairy': (255, 255, 100),     # Light yellow
                'fat': (100, 255, 255),       # Light cyan
                'other': (200, 200, 200)      # Light gray
            }
            
            # Draw segmentation regions
            for detection in detections:
                mask = detection['mask']
                class_name = detection['class_name']
                color = colors.get(class_name, (128, 128, 128))
                
                # Resize mask if needed
                if mask.shape != image_rgb.shape[:2]:
                    mask_resized = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask
                
                # Apply color to mask region
                seg_overlay[mask_resized > 0] = color
            
            # Blend depth map with segmentation
            vis_image = cv2.addWeighted(vis_image, 0.7, seg_overlay, 0.3, 0)
        
        # Add volume labels if available
        if volume_results:
            for result in volume_results:
                x1, y1, x2, y2 = result['bbox']
                food_type = result['food_type']
                volume = result['volume_cm3']
                weight = result['weight_grams']
                
                # Calculate center of bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Create volume label
                volume_label = f"{food_type}\n{volume:.1f}cm³\n{weight:.1f}g"
                
                # Draw label with background
                lines = volume_label.split('\n')
                line_height = 20
                max_width = 0
                
                # Calculate label background size
                for line in lines:
                    (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    max_width = max(max_width, w)
                
                # Draw background rectangle
                bg_y1 = center_y - len(lines) * line_height // 2 - 5
                bg_y2 = center_y + len(lines) * line_height // 2 + 5
                bg_x1 = center_x - max_width // 2 - 5
                bg_x2 = center_x + max_width // 2 + 5
                
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
                
                # Draw text lines
                for i, line in enumerate(lines):
                    text_y = center_y - len(lines) * line_height // 2 + i * line_height + 15
                    (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = center_x - w // 2
                    
                    cv2.putText(vis_image, line, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Add detection labels without volume
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                label = f"{class_name}\n{confidence:.3f}"
                lines = label.split('\n')
                
                # Draw background and text
                for i, line in enumerate(lines):
                    text_y = center_y + i * 20
                    (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = center_x - w // 2
                    
                    # Background
                    cv2.rectangle(vis_image, (text_x-5, text_y-15), (text_x+w+5, text_y+5), (0, 0, 0), -1)
                    
                    # Text
                    cv2.putText(vis_image, line, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
        
    except Exception as e:
        st.warning(f"Depth segmentation visualization failed: {e}")
        return None

# Configure page
st.set_page_config(
    page_title="Advanced Bring+Swiss Food Segmentation",
    page_icon="🍽️",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for SegFormer model
CONFIG = {
    'backbone': 'nvidia/mit-b3',
    'num_classes': 8,  # 7 food classes + background
    'classes': ['background', 'fruit', 'vegetable', 'carbohydrate', 'protein', 'dairy', 'fat', 'other'],
    'image_size': 512,
}

@st.cache_resource
def load_model():
    """Load the trained advanced bring+swiss segmentation model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model checkpoint paths - try different checkpoints
    checkpoint_dir = REPO_ROOT / "src" / "segmentation" / "mask_rcnn" / "training" / "checkpoints" / "advanced_bring_swiss_segmentation"
    
    checkpoint_options = {
        "Best mAP Model": checkpoint_dir / "segformer_best_map.pth",
        "Best Dice Model": checkpoint_dir / "segformer_best_dice.pth",
        "Final Model": checkpoint_dir / "segformer_final.pth"
    }
    
    # Try to load best mAP model first
    checkpoint_path = checkpoint_options["Best mAP Model"]
    
    if not checkpoint_path.exists():
        st.error(f"Model checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    try:
        # Create SegFormer model
        model = SegformerForSemanticSegmentation.from_pretrained(
            CONFIG['backbone'],
            num_labels=CONFIG['num_classes'],
            id2label={i: CONFIG['classes'][i] for i in range(CONFIG['num_classes'])},
            label2id={CONFIG['classes'][i]: i for i in range(CONFIG['num_classes'])},
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        # Load trained weights
        if checkpoint_path.suffix == '.pth' and 'checkpoint' not in checkpoint_path.name:
            # Load state dict only
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            # Load full checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        st.success(f"✅ Loaded Advanced Bring+Swiss SegFormer model from: {checkpoint_path.name}")
        st.info(f"📱 Device: {device}")
        st.info(f"🏷️ Classes: {CONFIG['classes']}")
        
        return model, device, CONFIG['classes']
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None

def get_transform():
    """Get preprocessing transforms for SegFormer."""
    return A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image, target_size=512):
    """Preprocess image for SegFormer model input."""
    # Convert to RGB if needed
    if isinstance(image, Image.Image):
        image_rgb = np.array(image)
    else:
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    
    if image_rgb.shape[-1] == 4:  # RGBA
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    
    # Apply transforms
    transform = get_transform()
    transformed = transform(image=image_rgb)
    tensor = transformed['image'].unsqueeze(0)
    
    return tensor, image_rgb

def predict_segmentation(model, device, tensor):
    """Run segmentation prediction with SegFormer."""
    tensor = tensor.to(device)
    
    with torch.no_grad():
        # SegFormer forward pass
        outputs = model(pixel_values=tensor)
        logits = outputs.logits
        
        # Resize logits to match input size if needed
        if logits.shape[-2:] != (CONFIG['image_size'], CONFIG['image_size']):
            logits = F.interpolate(
                logits, 
                size=(CONFIG['image_size'], CONFIG['image_size']), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get predicted classes
        pred_classes = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Get probability maps for each class
        prob_maps = probs.squeeze(0).cpu().numpy()
    
    return pred_classes, prob_maps

def analyze_prediction(pred_classes, prob_maps, class_names, confidence_threshold=0.3, original_shape=None):
    """Analyze prediction to find individual food items."""
    detections = []
    
    # Get scaling factors if original shape is provided
    pred_h, pred_w = pred_classes.shape
    if original_shape:
        orig_h, orig_w = original_shape[:2]
        scale_y = orig_h / pred_h
        scale_x = orig_w / pred_w
    else:
        scale_y = scale_x = 1.0
    
    # Process each non-background class
    for class_id in range(1, len(class_names)):
        class_name = class_names[class_id]
        
        # Get regions where this class has probability above threshold
        prob_mask = (prob_maps[class_id] >= confidence_threshold).astype(np.uint8)
        prob_pixels = prob_mask.sum()
        
        if prob_pixels == 0:
            continue
            
        # Find connected components
        labeled_mask, num_components = connected_components(prob_mask)
        
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            component_size = component_mask.sum()
            
            # Minimum size requirement
            if component_size < 100:
                continue
            
            # Get region properties
            props = regionprops(component_mask.astype(int))
            if not props:
                continue
                
            prop = props[0]
            
            # Calculate confidence (average probability in the region)
            region_prob = prob_maps[class_id][component_mask].mean()
            
            # Get bounding box and scale to original image coordinates
            y1, x1, y2, x2 = prop.bbox
            
            # Scale bounding box to original image size
            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y) 
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)
            
            area = component_mask.sum()
            
            detections.append({
                'class_name': class_name,
                'class_id': class_id,
                'confidence': region_prob,
                'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),  # Scaled to original image
                'bbox_pred': (x1, y1, x2, y2),  # Original prediction coordinates
                'area': area,
                'mask': component_mask,
                'detection_type': 'semantic_segmentation'
            })
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections

def create_visualization(image_rgb, pred_classes, detections, class_names):
    """Create visualization with segmentation detections."""
    # Create colormap for classes
    colors = {
        'fruit': (255, 0, 0),        # Red
        'vegetable': (0, 255, 0),    # Green
        'carbohydrate': (0, 0, 255), # Blue
        'protein': (255, 0, 255),    # Magenta
        'dairy': (255, 255, 0),      # Yellow
        'fat': (0, 255, 255),        # Cyan
        'other': (255, 255, 255)     # White
    }
    
    class_color_array = [
        [0, 0, 0],           # background - black
        [255, 0, 0],         # fruit - red
        [0, 255, 0],         # vegetable - green  
        [0, 0, 255],         # carbohydrate - blue
        [255, 0, 255],       # protein - magenta
        [255, 255, 0],       # dairy - yellow
        [0, 255, 255],       # fat - cyan
        [255, 255, 255],     # other - white
    ]
    
    # Resize pred_classes to match original image size
    orig_h, orig_w = image_rgb.shape[:2]
    if pred_classes.shape != (orig_h, orig_w):
        pred_classes_resized = cv2.resize(pred_classes.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        pred_classes_resized = pred_classes
    
    # Create segmentation overlay
    seg_colored = np.zeros_like(image_rgb)
    
    # Apply class colors
    for i, color in enumerate(class_color_array):
        seg_colored[pred_classes_resized == i] = color
    
    # Enhance detection regions
    for detection in detections:
        mask = detection['mask']
        class_name = detection['class_name']
        color = colors.get(class_name, (128, 128, 128))
        
        # Resize mask to match original image
        if mask.shape != (orig_h, orig_w):
            mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
            
        # Apply brighter color for detected regions
        seg_colored[mask_resized > 0] = [min(255, c + 50) for c in color]
    
    # Blend with original image
    overlay = cv2.addWeighted(image_rgb, 0.6, seg_colored, 0.4, 0)
    
    # Draw bounding boxes and labels
    vis_image = overlay.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get class-specific color
        box_color = colors.get(class_name, (255, 255, 255))
        
        # Vary thickness based on confidence
        thickness = max(1, int(confidence * 3))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, thickness)
        
        # Draw label with confidence
        label = f"{class_name}: {confidence:.3f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(vis_image, (x1, y1-20), (x1+w+5, y1), box_color, -1)
        cv2.putText(vis_image, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return vis_image, seg_colored

# Food volume estimation constants and functions
FOOD_DENSITY_MAP = {
    'fruit': 0.6,       # g/cm³ (average for fruits like apples, bananas)
    'vegetable': 0.5,   # g/cm³ (average for vegetables)
    'carbohydrate': 1.2, # g/cm³ (bread, rice, pasta)
    'protein': 1.0,     # g/cm³ (meat, fish)
    'dairy': 1.0,       # g/cm³ (milk products)
    'fat': 0.9,         # g/cm³ (oils, butter)
    'other': 0.8        # g/cm³ (mixed foods)
}

REFERENCE_OBJECT_SIZES = {
    'coin': {'diameter': 2.4, 'thickness': 0.2},  # US quarter in cm
    'phone': {'length': 15, 'width': 7.5},        # Average smartphone
    'hand': {'length': 18, 'width': 8.5},         # Average hand
    'plate': {'diameter': 25, 'height': 2}        # Standard dinner plate
}

@st.cache_resource
def load_utensil_detector():
    """Load the utensil detector for reference scaling."""
    if not REFERENCE_DETECTION_AVAILABLE:
        return None
    
    try:
        # Try to load YOLO model first
        yolo_model_path = REPO_ROOT / "src" / "reference" / "models" / "utensil_detector_yolo.pt"
        if yolo_model_path.exists():
            st.info(f"Found YOLO utensil model: {yolo_model_path}")
            # Use YOLO detector if available
            detector = YOLOUtensilDetector(str(yolo_model_path))
            return detector
        else:
            st.warning(f"YOLO model not found at: {yolo_model_path}")
            # Fallback to hybrid detector (which will use heuristic)
            detector = HybridUtensilDetector()
            return detector
    except Exception as e:
        st.warning(f"Could not load utensil detector: {e}")
        return None

def detect_reference_objects(image, detector):
    """Detect reference objects (utensils) in the image."""
    if detector is None or not REFERENCE_DETECTION_AVAILABLE:
        return []
    
    try:
        # Convert RGB to BGR for the detector
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detections = detector.detect(bgr_image)
        return detections
    except Exception as e:
        st.warning(f"Reference detection failed: {e}")
        return []

def estimate_pixel_to_cm_ratio(image, user_reference=None, reference_detections=None):
    """Estimate pixel to cm conversion ratio using reference objects or user input."""
    if user_reference and user_reference > 0:
        return user_reference
    
    # Try to use detected reference objects for accurate scaling
    if reference_detections:
        # Use the most confident detection
        best_detection = max(reference_detections, key=lambda d: d.confidence)
        scale_mm_per_pixel = best_detection.scale_mm_per_pixel
        scale_cm_per_pixel = scale_mm_per_pixel / 10.0  # Convert mm to cm
        return scale_cm_per_pixel
    
    # Default estimation based on image size (rough approximation)
    height, width = image.shape[:2]
    # Assume average plate size covers ~60% of image width
    estimated_plate_width_pixels = width * 0.6
    estimated_plate_width_cm = 25  # Standard plate diameter
    
    return estimated_plate_width_cm / estimated_plate_width_pixels

def calculate_food_volume_ellipsoid(mask, pixel_to_cm):
    """Calculate food volume using ellipsoid approximation."""
    # Get region properties
    props = regionprops(mask.astype(int))
    if not props:
        return 0.0
    
    prop = props[0]
    
    # Get dimensions in pixels
    height_px, width_px = prop.bbox[2] - prop.bbox[0], prop.bbox[3] - prop.bbox[1]
    
    # Convert to cm
    height_cm = height_px * pixel_to_cm
    width_cm = width_px * pixel_to_cm
    
    # Estimate depth (assume roughly circular cross-section, depth = 60% of width)
    depth_cm = width_cm * 0.6
    
    # Calculate ellipsoid volume: (4/3) * π * a * b * c
    volume_cm3 = (4/3) * math.pi * (width_cm/2) * (height_cm/2) * (depth_cm/2)
    
    # Account for food compactness (foods are not perfect ellipsoids)
    compactness_factor = prop.filled_area / (prop.bbox[2] - prop.bbox[0]) / (prop.bbox[3] - prop.bbox[1])
    
    return volume_cm3 * compactness_factor

def calculate_food_volume_depth_estimation(mask, pixel_to_cm, food_type):
    """Calculate volume using depth estimation based on food type."""
    # Get mask area in pixels
    area_pixels = np.sum(mask > 0)
    area_cm2 = area_pixels * (pixel_to_cm ** 2)
    
    # Estimate depth based on food type
    depth_factors = {
        'fruit': 0.8,       # Fruits are roughly spherical
        'vegetable': 0.6,   # Vegetables vary but often flatter
        'carbohydrate': 0.3, # Rice, pasta often shallow
        'protein': 0.4,     # Meat pieces moderate thickness
        'dairy': 0.2,       # Cheese, milk products thin
        'fat': 0.1,         # Spreads, oils very thin
        'other': 0.5        # Average thickness
    }
    
    # Calculate equivalent diameter from area
    equivalent_diameter_cm = 2 * math.sqrt(area_cm2 / math.pi)
    estimated_depth_cm = equivalent_diameter_cm * depth_factors.get(food_type, 0.5)
    
    # Calculate volume (area × average depth)
    volume_cm3 = area_cm2 * estimated_depth_cm
    
    return volume_cm3

def estimate_food_weight(volume_cm3, food_type):
    """Estimate food weight from volume and density."""
    density = FOOD_DENSITY_MAP.get(food_type, 0.8)
    weight_grams = volume_cm3 * density
    return weight_grams

def analyze_food_volumes(detections, image, pixel_to_cm, volume_method='ellipsoid'):
    """Analyze volumes for all detected food items."""
    volume_results = []
    
    for detection in detections:
        food_type = detection['class_name']
        mask = detection['mask']
        
        # Calculate volume based on selected method
        if volume_method == 'ellipsoid':
            volume = calculate_food_volume_ellipsoid(mask, pixel_to_cm)
        else:
            volume = calculate_food_volume_depth_estimation(mask, pixel_to_cm, food_type)
        
        # Estimate weight
        weight = estimate_food_weight(volume, food_type)
        
        volume_results.append({
            'food_type': food_type,
            'confidence': detection['confidence'],
            'volume_cm3': volume,
            'weight_grams': weight,
            'area_pixels': np.sum(mask > 0),
            'bbox': detection['bbox']
        })
    
    return volume_results

def create_reference_visualization(image, reference_detections):
    """Create visualization showing detected reference objects."""
    vis_image = image.copy()
    
    # Get image dimensions for validation
    img_height, img_width = vis_image.shape[:2]
    
    for ref_det in reference_detections:
        x1, y1, x2, y2 = ref_det.bbox
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # Make sure we have a valid rectangle
        if x2 > x1 and y2 > y1:
            # Draw bounding box in bright green for reference objects
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Create label with utensil info and coordinates for debugging
            label = f"{ref_det.object_type}\n{ref_det.scale_mm_per_pixel:.2f}mm/px\nConf: {ref_det.confidence:.3f}\nBox: ({x1},{y1},{x2},{y2})"
            
            # Draw multi-line label
            lines = label.split('\n')
            line_height = 15
            y_offset = y1 - len(lines) * line_height - 5
            
            for i, line in enumerate(lines):
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # Background rectangle for text
                cv2.rectangle(vis_image, (x1, y_offset + i * line_height - 12), 
                             (x1 + w + 5, y_offset + i * line_height + 3), (0, 255, 0), -1)
                # Text
                cv2.putText(vis_image, line, (x1 + 2, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return vis_image

def create_volume_visualization(image, volume_results, pixel_to_cm):
    """Create visualization with volume information."""
    vis_image = image.copy()
    
    for result in volume_results:
        x1, y1, x2, y2 = result['bbox']
        food_type = result['food_type']
        volume = result['volume_cm3']
        weight = result['weight_grams']
        
        # Color for different food types
        colors = {
            'fruit': (255, 0, 0),        # Red
            'vegetable': (0, 255, 0),    # Green
            'carbohydrate': (0, 0, 255), # Blue
            'protein': (255, 0, 255),    # Magenta
            'dairy': (255, 255, 0),      # Yellow
            'fat': (0, 255, 255),        # Cyan
            'other': (255, 255, 255)     # White
        }
        
        color = colors.get(food_type, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Create label with volume info
        label = f"{food_type}\nVol: {volume:.1f}cm³\nWeight: {weight:.1f}g"
        
        # Draw multi-line label
        lines = label.split('\n')
        line_height = 15
        y_offset = y1 - len(lines) * line_height - 5
        
        for i, line in enumerate(lines):
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Background rectangle for text
            cv2.rectangle(vis_image, (x1, y_offset + i * line_height - 12), 
                         (x1 + w + 5, y_offset + i * line_height + 3), color, -1)
            # Text
            cv2.putText(vis_image, line, (x1 + 2, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return vis_image

def main():
    st.title("🍽️ Advanced Bring+Swiss Food Segmentation with Volume Detection")
    st.markdown("**SegFormer Model** (MiT-B3 backbone) - Combined Bring+Swiss Dataset + Volume Estimation + Reference Scaling")
    
    # Load model and utensil detector
    model, device, class_names = load_model()
    if model is None:
        return
    
    utensil_detector = load_utensil_detector()
    if utensil_detector and REFERENCE_DETECTION_AVAILABLE:
        st.success("✅ Reference utensil detector loaded for accurate scaling")
    else:
        st.info("ℹ️ Using manual scaling (no utensil detector available)")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Minimum confidence for food detection"
        )
        
        st.header("📏 Volume Detection")
        
        enable_volume = st.checkbox(
            "Enable Volume Estimation",
            value=True,
            help="Calculate estimated volume and weight of detected foods"
        )
        
        if enable_volume:
            volume_method = st.selectbox(
                "Volume Calculation Method",
                ["ellipsoid", "depth_estimation"],
                index=0,
                help="Method for calculating food volume"
            )
            
            use_reference_detection = st.checkbox(
                "Auto-detect Reference Objects",
                value=REFERENCE_DETECTION_AVAILABLE,
                disabled=not REFERENCE_DETECTION_AVAILABLE,
                help="Automatically detect utensils for accurate scaling"
            )
            
            pixel_to_cm = st.number_input(
                "Manual Pixel to CM Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.3f",
                help="Manual conversion factor (0 = auto-detect using references or estimation)"
            )
            
            show_reference_guide = st.checkbox(
                "Show Reference Size Guide",
                value=False,
                help="Display common reference object sizes"
            )
            
            if show_reference_guide:
                st.markdown("**Reference Objects:**")
                st.markdown("🪙 **Coin (Quarter)**: 2.4cm diameter")
                st.markdown("📱 **Phone**: ~15cm × 7.5cm") 
                st.markdown("✋ **Hand**: ~18cm × 8.5cm")
                st.markdown("🍽️ **Plate**: ~25cm diameter")
        
        st.header("🎨 Food Classes")
        st.markdown("🔴 **Red**: Fruit")  
        st.markdown("🟢 **Green**: Vegetable")
        st.markdown("🔵 **Blue**: Carbohydrate")
        st.markdown("🟣 **Magenta**: Protein")
        st.markdown("🟡 **Yellow**: Dairy")
        st.markdown("🔵 **Cyan**: Fat")
        st.markdown("⚪ **White**: Other")
        
        st.header("ℹ️ Model Info")
        st.write(f"**Architecture**: SegFormer + MiT-B3")
        st.write(f"**Training**: Combined Bring+Swiss Dataset")
        st.write(f"**Classes**: {len(class_names)-1} + background")
        st.write(f"**Loss**: Combined Focal + Dice Loss")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Food Image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to test the 7-class segmentation model"
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.markdown(f"## 📸 Testing: {uploaded_file.name}")
        
        # Preprocess
        with st.spinner("Preprocessing image..."):
            tensor, image_rgb = preprocess_image(image)
            
        # Run prediction
        with st.spinner("Running 7-class segmentation..."):
            pred_classes, prob_maps = predict_segmentation(model, device, tensor)
            
        # Analyze prediction
        detections = analyze_prediction(pred_classes, prob_maps, class_names, confidence_threshold, image_rgb.shape)
        
        # Reference object detection for accurate scaling
        reference_detections = []
        if enable_volume and use_reference_detection and utensil_detector:
            with st.spinner("Detecting reference objects..."):
                reference_detections = detect_reference_objects(image_rgb, utensil_detector)
        
        # Volume analysis if enabled
        volume_results = []
        if enable_volume and detections:
            # Auto-estimate pixel_to_cm if not provided, using reference detections if available
            if pixel_to_cm == 0:
                pixel_to_cm = estimate_pixel_to_cm_ratio(image_rgb, None, reference_detections)
            
            volume_results = analyze_food_volumes(detections, image_rgb, pixel_to_cm, volume_method)
        
        # Display results
        st.markdown(f"### 🔍 Results")
        
        # Show reference detection results
        if reference_detections:
            st.success(f"✅ Detected {len(reference_detections)} reference objects for accurate scaling!")
            for ref_det in reference_detections:
                st.info(f"📏 {ref_det.object_type}: {ref_det.scale_mm_per_pixel:.3f} mm/pixel (confidence: {ref_det.confidence:.3f})")
        elif enable_volume and use_reference_detection:
            st.warning("⚠️ No reference objects detected. Using estimated scaling.")
        
        if detections:
            st.success(f"Detected {len(detections)} food items!")
            
            if enable_volume and volume_results:
                # Show detection summary with volume
                total_volume = 0
                total_weight = 0
                
                for i, (det, vol_result) in enumerate(zip(detections, volume_results), 1):
                    total_volume += vol_result['volume_cm3']
                    total_weight += vol_result['weight_grams']
                    
                    st.markdown(
                        f"**{i}. {det['class_name'].title()}** - "
                        f"Confidence: {det['confidence']:.3f} - "
                        f"Volume: {vol_result['volume_cm3']:.1f}cm³ - "
                        f"Weight: {vol_result['weight_grams']:.1f}g"
                    )
                
                # Show totals
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Items", len(detections))
                with col2:
                    st.metric("Total Volume", f"{total_volume:.1f} cm³")
                with col3:
                    st.metric("Total Weight", f"{total_weight:.1f} g")
            else:
                # Show detection summary without volume
                for i, det in enumerate(detections, 1):
                    st.markdown(f"**{i}. {det['class_name'].title()}** - Confidence: {det['confidence']:.3f} - Area: {det['area']} pixels")
        else:
            st.warning(f"No food items detected above confidence threshold {confidence_threshold:.3f}")
            st.info("Try lowering the confidence threshold.")
        
        # Create visualization - always create seg_colored for mask display
        vis_image, seg_colored = create_visualization(image_rgb, pred_classes, detections, class_names)
        
        # Add mask overlay option in sidebar
        show_mask_overlay = st.sidebar.checkbox("Show Mask Overlay", value=True, help="Overlay segmentation masks on the result image")
        overlay_alpha = st.sidebar.slider("Mask Transparency", 0.0, 1.0, 0.4, 0.1, 
                                        help="0.0 = transparent, 1.0 = opaque")
        
        # Always apply mask overlay to the base visualization
        if show_mask_overlay:
            vis_image = cv2.addWeighted(vis_image, 1.0 - overlay_alpha, seg_colored, overlay_alpha, 0)
        
        if enable_volume and volume_results:
            # Create volume-enhanced visualization
            volume_vis = create_volume_visualization(image_rgb, volume_results, pixel_to_cm)
            vis_title = "7-Class Segmentation + Volume Detection"
            
            # Apply mask overlay to volume visualization too
            if show_mask_overlay:
                vis_image = cv2.addWeighted(volume_vis, 1.0 - overlay_alpha, seg_colored, overlay_alpha, 0)
            else:
                vis_image = volume_vis
        else:
            vis_title = "7-Class Segmentation Result"
        
        # Add reference detections to visualization if available
        if reference_detections:
            vis_image = create_reference_visualization(vis_image, reference_detections)
            if "Reference" not in vis_title:
                vis_title += " + Reference Detection"
        
        # Update title if mask overlay is enabled
        if show_mask_overlay:
            if "Mask Overlay" not in vis_title:
                vis_title += " + Mask Overlay"
        
        # Display images
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image_rgb, use_container_width=True)
            
        with col2:
            st.markdown(f"#### {vis_title}")
            st.image(vis_image, use_container_width=True)
        
        # Generate depth map and depth+volume visualization
        if st.sidebar.checkbox("Show Depth Analysis", value=True, help="Generate depth map and volume analysis"):
            st.markdown("---")
            st.markdown("### 📐 Depth Analysis")
            
            # Generate depth map
            depth_map = generate_depth_map(image_rgb)
            
            # Create depth+segmentation visualization
            depth_seg_image = create_depth_segmentation_visualization(image_rgb, depth_map, detections, volume_results if enable_volume else None)
            
            # Display depth images
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Depth Map")
                if depth_map is not None:
                    st.image(depth_map, use_container_width=True, caption="Estimated depth (darker = closer)")
                else:
                    st.error("Failed to generate depth map")
            
            with col4:
                st.markdown("#### Segmentation + Depth + Volume")
                if depth_seg_image is not None:
                    st.image(depth_seg_image, use_container_width=True, caption="Food segments with depth and volume labels")
                else:
                    st.error("Failed to generate depth segmentation visualization")
        
        # Volume-specific visualizations
        if enable_volume and volume_results:
            st.markdown("### 📊 Volume Analysis")
            
            # Create volume distribution chart
            food_types = [result['food_type'] for result in volume_results]
            volumes = [result['volume_cm3'] for result in volume_results]
            weights = [result['weight_grams'] for result in volume_results]
            
            if len(volume_results) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Volume Distribution (cm³)**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.bar(food_types, volumes, 
                                 color=['red', 'green', 'blue', 'magenta', 'yellow', 'cyan', 'white'][:len(food_types)])
                    ax.set_ylabel('Volume (cm³)')
                    ax.set_title('Food Volume Distribution')
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar, volume in zip(bars, volumes):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{volume:.1f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    st.markdown("**Weight Distribution (g)**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.bar(food_types, weights,
                                 color=['red', 'green', 'blue', 'magenta', 'yellow', 'cyan', 'white'][:len(food_types)])
                    ax.set_ylabel('Weight (g)')
                    ax.set_title('Food Weight Distribution')
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar, weight in zip(bars, weights):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{weight:.1f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Nutrition estimates (basic)
            st.markdown("### 🥗 Estimated Nutrition Information")
            st.info("⚠️ These are rough estimates based on average food densities. For accurate nutrition info, use actual measurements and food databases.")
            
            nutrition_estimates = {
                'fruit': {'calories_per_g': 0.5, 'protein_per_g': 0.01, 'carbs_per_g': 0.12},
                'vegetable': {'calories_per_g': 0.25, 'protein_per_g': 0.02, 'carbs_per_g': 0.05},
                'carbohydrate': {'calories_per_g': 3.5, 'protein_per_g': 0.08, 'carbs_per_g': 0.75},
                'protein': {'calories_per_g': 2.5, 'protein_per_g': 0.25, 'carbs_per_g': 0.0},
                'dairy': {'calories_per_g': 1.5, 'protein_per_g': 0.08, 'carbs_per_g': 0.05},
                'fat': {'calories_per_g': 9.0, 'protein_per_g': 0.0, 'carbs_per_g': 0.0},
                'other': {'calories_per_g': 2.0, 'protein_per_g': 0.05, 'carbs_per_g': 0.3}
            }
            
            total_calories = 0
            total_protein = 0
            total_carbs = 0
            
            for result in volume_results:
                food_type = result['food_type']
                weight = result['weight_grams']
                nutrition = nutrition_estimates.get(food_type, nutrition_estimates['other'])
                
                total_calories += weight * nutrition['calories_per_g']
                total_protein += weight * nutrition['protein_per_g']
                total_carbs += weight * nutrition['carbs_per_g']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Est. Calories", f"{total_calories:.0f} kcal")
            with col2:
                st.metric("Est. Protein", f"{total_protein:.1f} g")
            with col3:
                st.metric("Est. Carbs", f"{total_carbs:.1f} g")
        
        # Raw segmentation masks
        if st.checkbox("Show Raw Segmentation Masks"):
            st.markdown("#### Segmentation Masks")
            if 'seg_colored' in locals():
                st.image(seg_colored, use_container_width=True)
            else:
                st.error("Segmentation masks not available")
        
        
        # Individual masks
        if st.checkbox("Show Individual Masks") and detections:
            st.markdown("#### Individual Detection Masks")
            
            # Option to show masks with bounding boxes
            show_bbox_on_masks = st.checkbox("Show bounding boxes on masks")
            
            cols = st.columns(min(len(detections), 4))
            
            for i, detection in enumerate(detections):
                with cols[i % len(cols)]:
                    st.markdown(f"**{detection['class_name']}**")
                    
                    if show_bbox_on_masks:
                        # Create mask with bounding box overlay
                        mask_rgb = np.stack([detection['mask'] * 255] * 3, axis=-1).astype(np.uint8)
                        x1, y1, x2, y2 = detection['bbox']
                        
                        # Scale bounding box to mask coordinates if needed
                        mask_h, mask_w = detection['mask'].shape
                        orig_h, orig_w = image_rgb.shape[:2]
                        
                        if mask_h != orig_h or mask_w != orig_w:
                            # Scale bbox coordinates to mask size
                            scale_x = mask_w / orig_w
                            scale_y = mask_h / orig_h
                            x1_mask = int(x1 * scale_x)
                            y1_mask = int(y1 * scale_y)
                            x2_mask = int(x2 * scale_x)
                            y2_mask = int(y2 * scale_y)
                        else:
                            x1_mask, y1_mask, x2_mask, y2_mask = x1, y1, x2, y2
                        
                        # Draw bounding box on mask
                        cv2.rectangle(mask_rgb, (x1_mask, y1_mask), (x2_mask, y2_mask), (0, 255, 0), 2)
                        st.image(mask_rgb, use_container_width=True)
                        st.caption(f"Conf: {detection['confidence']:.3f} | BBox: ({x1},{y1},{x2},{y2})")
                    else:
                        # Show regular mask
                        mask_vis = detection['mask'] * 255
                        st.image(mask_vis, use_container_width=True)
                        st.caption(f"Conf: {detection['confidence']:.3f}")
        
        # Show debug info
        with st.expander("🔧 Debug Information"):
            unique_classes = np.unique(pred_classes)
            max_probs = [prob_maps[i].max() for i in range(len(class_names))]
            
            st.markdown("**Predicted Classes:**")
            st.write(f"Unique classes in prediction: {unique_classes}")
            
            st.markdown("**Class Probabilities:**")
            for i, (class_name, max_prob) in enumerate(zip(class_names, max_probs)):
                st.write(f"- {class_name}: {max_prob:.4f}")
        
        # Raw predictions data
        with st.expander("Raw Predictions Data"):
            prediction_data = {
                "num_detections": len(detections),
                "confidence_threshold": confidence_threshold,
                "detections": [
                    {
                        "class": det['class_name'],
                        "confidence": det['confidence'],
                        "bbox": [int(x) for x in det['bbox']],
                        "area": int(det['area'])
                    }
                    for det in detections
                ]
            }
            
            if enable_volume and volume_results:
                prediction_data["volume_analysis"] = {
                    "pixel_to_cm_ratio": pixel_to_cm,
                    "volume_method": volume_method,
                    "volume_results": [
                        {
                            "food_type": result['food_type'],
                            "volume_cm3": round(result['volume_cm3'], 2),
                            "weight_grams": round(result['weight_grams'], 2),
                            "confidence": result['confidence']
                        }
                        for result in volume_results
                    ]
                }
            
            st.json(prediction_data)
    
    else:
        st.info("👆 Upload a food image to test the 7-class segmentation model")
        
        # Show model info
        st.markdown("### 📊 Model Information")
        st.markdown(f"""
        - **Architecture**: SegFormer with MiT-B3 transformer backbone
        - **Dataset**: Combined Bring+Swiss Dataset (7 classes)
        - **Training**: 10 epochs with advanced augmentation
        - **Loss Function**: Combined Focal + Dice Loss
        - **Input Size**: 512×512 pixels
        - **Classes**: fruit, vegetable, carbohydrate, protein, dairy, fat, other
        """)

if __name__ == "__main__":
    main()