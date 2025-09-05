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
from transformers import SegformerForSemanticSegmentation
from scipy.ndimage import label as connected_components
from skimage.measure import regionprops

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

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

def analyze_prediction(pred_classes, prob_maps, class_names, confidence_threshold=0.3):
    """Analyze prediction to find individual food items."""
    detections = []
    
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
            
            # Get bounding box
            y1, x1, y2, x2 = prop.bbox
            area = component_mask.sum()
            
            detections.append({
                'class_name': class_name,
                'class_id': class_id,
                'confidence': region_prob,
                'bbox': (x1, y1, x2, y2),
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

def main():
    st.title("🍽️ Advanced Bring+Swiss Food Segmentation")
    st.markdown("**SegFormer Model** (MiT-B3 backbone) - Combined Bring+Swiss Dataset")
    
    # Load model
    model, device, class_names = load_model()
    if model is None:
        return
    
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
        
        st.markdown("**7 Food Classes:**")
        st.markdown("🔴 **Red**: Fruit")  
        st.markdown("🟢 **Green**: Vegetable")
        st.markdown("🔵 **Blue**: Carbohydrate")
        st.markdown("🟣 **Magenta**: Protein")
        st.markdown("🟡 **Yellow**: Dairy")
        st.markdown("🔵 **Cyan**: Fat")
        st.markdown("⚪ **White**: Other")
        
        st.markdown("**Model Info:**")
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
        detections = analyze_prediction(pred_classes, prob_maps, class_names, confidence_threshold)
        
        # Display results
        st.markdown(f"### 🔍 Results")
        
        if detections:
            st.success(f"Detected {len(detections)} food items!")
            
            # Show detection summary
            for i, det in enumerate(detections, 1):
                st.markdown(f"**{i}. {det['class_name'].title()}** - Confidence: {det['confidence']:.3f} - Area: {det['area']} pixels")
        else:
            st.warning(f"No food items detected above confidence threshold {confidence_threshold:.3f}")
            st.info("Try lowering the confidence threshold.")
        
        # Create visualization
        vis_image, seg_colored = create_visualization(image_rgb, pred_classes, detections, class_names)
        
        # Display images
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image_rgb, use_container_width=True)
            
        with col2:
            st.markdown("#### 7-Class Segmentation Result")
            st.image(vis_image, use_container_width=True)
        
        # Raw segmentation masks
        if st.checkbox("Show Raw Segmentation Masks"):
            st.markdown("#### Segmentation Masks")
            st.image(seg_colored, use_container_width=True)
        
        # Individual masks
        if st.checkbox("Show Individual Masks") and detections:
            st.markdown("#### Individual Detection Masks")
            cols = st.columns(min(len(detections), 4))
            
            for i, detection in enumerate(detections):
                with cols[i % len(cols)]:
                    st.markdown(f"**{detection['class_name']}**")
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
            st.json({
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
            })
    
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