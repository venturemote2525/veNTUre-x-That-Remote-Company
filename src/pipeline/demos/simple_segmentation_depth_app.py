#!/usr/bin/env python3
"""
Simple Segmentation + Depth Integration

Combines food segmentation (Mask R-CNN) and depth estimation (Depth Anything)
without the full nutrition analysis pipeline. Focus on core integration.

Launch:
  python -m streamlit run src/nutrition_analysis/simple_segmentation_depth_app.py
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F

# Add model paths to system path
REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTH_PATH = REPO_ROOT / "src" / "models" / "depth_anything"
MASK_RCNN_PATH = REPO_ROOT / "src" / "models" / "pytorch_mask_rcnn"

sys.path.append(str(DEPTH_PATH))
sys.path.append(str(MASK_RCNN_PATH))
sys.path.append(str(MASK_RCNN_PATH / "utils"))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure page
st.set_page_config(
    page_title="Food Segmentation + Depth Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib backend for streamlit
plt.switch_backend('Agg')


def safe_import_depth_anything():
    """Safely import Depth Anything modules with error handling."""
    try:
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        return DepthAnything, Compose, Resize, NormalizeImage, PrepareForNet
    except ImportError as e:
        st.error(f"Could not import Depth Anything modules: {e}")
        st.info("Install missing dependencies: `pip install transformers huggingface-hub timm`")
        return None, None, None, None, None


def safe_import_mask_rcnn():
    """Safely import Mask R-CNN modules with error handling."""
    try:
        from train_maskrcnn_food import FoodMaskRCNNTrainer
        from utils.config import FoodMaskRCNNConfig
        return FoodMaskRCNNTrainer, FoodMaskRCNNConfig
    except ImportError as e:
        st.error(f"Could not import Mask R-CNN modules: {e}")
        st.info("Make sure your Mask R-CNN training code is available")
        return None, None


def list_checkpoint_dirs():
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


def pick_checkpoint(ckpt_dir: Path):
    """Find the best checkpoint in a directory."""
    if (ckpt_dir / "best_checkpoint.pth").exists():
        return ckpt_dir / "best_checkpoint.pth"
    if (ckpt_dir / "latest_checkpoint.pth").exists():
        return ckpt_dir / "latest_checkpoint.pth"
    epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    return epoch_ckpts[-1] if epoch_ckpts else None


@st.cache_resource(show_spinner=False)
def load_segmentation_model(checkpoint_path: str):
    """Load and cache Mask R-CNN segmentation model."""
    FoodMaskRCNNTrainer, FoodMaskRCNNConfig = safe_import_mask_rcnn()
    
    if FoodMaskRCNNTrainer is None:
        return None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg_dict = checkpoint.get("config", {})
    class_names = cfg_dict.get("class_names", ["fruit", "vegetable", "carbohydrate", "protein", "dairy", "fat", "other"])

    cfg = FoodMaskRCNNConfig(
        name=cfg_dict.get("name", "segmentation_depth_integration"),
        class_names=class_names,
        backbone=cfg_dict.get("backbone", "resnet50"),
        epochs=1,
        batch_size=1,
        image_min_size=cfg_dict.get("image_min_size", 640),
        image_max_size=cfg_dict.get("image_max_size", 800),
        checkpoint_dir=str(Path(checkpoint_path).parent),
        log_dir=str(Path(checkpoint_path).parent.parent / "logs")
    )

    trainer = FoodMaskRCNNTrainer(cfg)
    trainer.build_model()
    state = checkpoint.get("model_state_dict") or checkpoint
    trainer.model.load_state_dict(state)
    trainer.model.eval().to(device)

    return trainer.model, device, class_names


@st.cache_resource(show_spinner=False)
def load_depth_model(encoder: str = "vitb"):
    """Load and cache Depth Anything model."""
    DepthAnything, Compose, Resize, NormalizeImage, PrepareForNet = safe_import_depth_anything()
    
    if DepthAnything is None:
        return None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = f"LiheYoung/depth_anything_{encoder}14"
    
    try:
        model = DepthAnything.from_pretrained(model_id).to(device).eval()
        
        transform = Compose([
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
        
        return model, device, transform
    except Exception as e:
        st.error(f"Failed to load Depth Anything model: {e}")
        return None, None, None


def tensor_from_image(image: np.ndarray):
    """Convert image to tensor for Mask R-CNN."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor


def run_segmentation(model, device, image: np.ndarray, conf_threshold: float = 0.5):
    """Run food segmentation on image."""
    inp = tensor_from_image(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(inp)
    
    pred = predictions[0]
    boxes = pred.get("boxes", torch.empty(0, 4))
    scores = pred.get("scores", torch.empty(0))
    labels = pred.get("labels", torch.empty(0))
    masks = pred.get("masks", torch.empty(0, 0, 0))
    
    if boxes.numel() == 0:
        return [], [], [], []
    
    # Convert to numpy and filter by confidence
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Process masks
    processed_masks = []
    if len(masks) > 0:
        for i, keep_idx in enumerate(np.where(keep)[0]):
            if keep_idx < len(masks):
                mask = masks[keep_idx].detach().cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                processed_masks.append(mask)
            else:
                # Create mask from bounding box if no mask available
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = boxes[i].astype(int)
                mask[y1:y2, x1:x2] = 1
                processed_masks.append(mask)
    
    return boxes, scores, labels, processed_masks


def run_depth_estimation(model, device, transform, image: np.ndarray):
    """Run depth estimation on image."""
    h, w = image.shape[:2]
    
    # Preprocess
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    net_input = transform({"image": rgb})["image"]
    net_input = torch.from_numpy(net_input).unsqueeze(0).to(device)
    
    # Estimate depth
    with torch.no_grad():
        depth = model(net_input)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.cpu().numpy()
    
    return depth


def colorize_depth(depth: np.ndarray, cmap: str = "INFERNO"):
    """Colorize depth map."""
    depth = depth.astype(np.float32)
    dmin, dmax = float(depth.min()), float(depth.max())
    
    if dmax - dmin < 1e-6:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

    cmaps = {
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "TURBO": cv2.COLORMAP_TURBO,
        "JET": cv2.COLORMAP_JET,
    }
    
    cv2_cmap = cmaps.get(cmap.upper(), cv2.COLORMAP_INFERNO)
    colored = cv2.applyColorMap(norm, cv2_cmap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def draw_segmentation_results(image: np.ndarray, boxes, scores, labels, masks, class_names):
    """Draw segmentation results on image."""
    result_image = image.copy()
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get class name
        class_idx = int(label) - 1  # Labels are 1-indexed
        if 0 <= class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"class_{label}"
        
        # Draw label
        label_text = f"{class_name}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_image, (x1, y1-20), (x1+w, y1), (0, 255, 0), -1)
        cv2.putText(result_image, label_text, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw mask overlay
        if i < len(masks):
            mask = masks[i]
            colored_mask = np.zeros_like(result_image)
            colored_mask[:, :, 2] = 255  # Red channel
            mask_3d = np.dstack([mask]*3)
            result_image = np.where(mask_3d, 
                                   cv2.addWeighted(colored_mask, 0.3, result_image, 0.7, 0), 
                                   result_image)
    
    return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)


def analyze_depth_per_food_item(depth_map, masks, boxes, class_names, labels):
    """Analyze depth information for each detected food item."""
    analysis_results = []
    
    for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
        if mask.sum() == 0:  # Skip empty masks
            continue
        
        # Extract depth values within mask
        masked_depth = depth_map[mask > 0]
        
        if len(masked_depth) == 0:
            continue
        
        # Get class name
        class_idx = int(label) - 1
        if 0 <= class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"class_{label}"
        
        # Calculate statistics
        result = {
            'item_id': i,
            'class_name': class_name,
            'mask_area': int(mask.sum()),
            'depth_min': float(masked_depth.min()),
            'depth_max': float(masked_depth.max()),
            'depth_mean': float(masked_depth.mean()),
            'depth_std': float(masked_depth.std()),
            'bbox': box.tolist()
        }
        
        analysis_results.append(result)
    
    return analysis_results


def main():
    """Main Streamlit application."""
    st.title("🍎 Food Segmentation + Depth Analysis")
    st.markdown("Combines Mask R-CNN food segmentation with Depth Anything estimation")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        depth_encoder = st.selectbox(
            "Depth Encoder",
            options=["vits", "vitb", "vitl"],
            index=1,
            help="Depth Anything encoder size"
        )
        
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        colormap = st.selectbox(
            "Depth Colormap",
            options=["INFERNO", "MAGMA", "PLASMA", "VIRIDIS", "TURBO", "JET"],
            index=0
        )
        
        st.subheader("Display Options")
        show_original = st.checkbox("Show Original Image", value=True)
        show_segmentation = st.checkbox("Show Segmentation Results", value=True)
        show_depth = st.checkbox("Show Depth Map", value=True)
        show_analysis = st.checkbox("Show Depth Analysis per Food Item", value=True)
    
    # Main content area
    uploaded_files = st.file_uploader(
        "Upload Food Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more images of food for analysis"
    )
    
    if not uploaded_files:
        st.info("👆 Upload food images to begin analysis")
        
        # Show information about the integration
        st.markdown("### 🔬 About This Integration")
        st.markdown("""
        This app combines two powerful models:
        
        - **🎯 Mask R-CNN**: Detects and segments food items into 7 categories
          - Fruit, Vegetable, Carbohydrate, Protein, Dairy, Fat, Other
          
        - **📏 Depth Anything**: Estimates per-pixel depth information
          - Provides 3D understanding of food volume and structure
          
        **Integration Features:**
        - Analyze depth information for each detected food item
        - Calculate volume estimates using depth + segmentation
        - Visualize both segmentation masks and depth maps
        """)
        return
    
    # Check for available checkpoints
    ckpt_dirs = list_checkpoint_dirs()
    if not ckpt_dirs:
        st.error("No segmentation model checkpoints found. Please train a model first.")
        return
    
    # Select checkpoint
    default_idx = 0
    for i, d in enumerate(ckpt_dirs):
        if "resnet50" in d.name:
            default_idx = i
            break
    
    with st.sidebar:
        st.subheader("Model Checkpoint")
        selected_dir = st.selectbox(
            "Checkpoint Directory", 
            options=[str(d) for d in ckpt_dirs], 
            index=default_idx
        )
    
    ckpt_dir = Path(selected_dir)
    ckpt_path = pick_checkpoint(ckpt_dir)
    
    if not ckpt_path:
        st.error("No checkpoint files found in the selected directory.")
        return
    
    # Load models
    with st.spinner("Loading models..."):
        # Load segmentation model
        seg_model, seg_device, class_names = load_segmentation_model(str(ckpt_path))
        if seg_model is None:
            st.error("Failed to load segmentation model")
            return
        
        # Load depth model
        depth_model, depth_device, depth_transform = load_depth_model(depth_encoder)
        if depth_model is None:
            st.error("Failed to load depth model")
            return
    
    st.success(f"✅ Models loaded successfully! Using {seg_device.upper()} device")
    st.info(f"Segmentation classes: {', '.join(class_names)}")
    
    # Process uploaded images
    for i, uploaded_file in enumerate(uploaded_files, 1):
        st.markdown(f"## 📸 Image {i}: {uploaded_file.name}")
        
        # Read image
        try:
            file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error(f"Could not read image: {uploaded_file.name}")
                continue
                
        except Exception as e:
            st.error(f"Error loading image {uploaded_file.name}: {e}")
            continue
        
        # Run analysis
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            # Run segmentation
            boxes, scores, labels, masks = run_segmentation(
                seg_model, seg_device, image, confidence_threshold
            )
            
            # Run depth estimation
            depth_map = run_depth_estimation(depth_model, depth_device, depth_transform, image)
        
        # Display results
        if show_original or show_segmentation or show_depth:
            cols = []
            if show_original:
                cols.append("Original")
            if show_segmentation:
                cols.append("Segmentation")
            if show_depth:
                cols.append("Depth")
            
            display_cols = st.columns(len(cols))
            
            col_idx = 0
            if show_original:
                with display_cols[col_idx]:
                    st.markdown("#### Original Image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                col_idx += 1
            
            if show_segmentation:
                with display_cols[col_idx]:
                    st.markdown("#### Segmentation Results")
                    if len(boxes) > 0:
                        seg_result = draw_segmentation_results(
                            image, boxes, scores, labels, masks, class_names
                        )
                        st.image(seg_result, use_column_width=True)
                        st.caption(f"Detected {len(boxes)} food items")
                    else:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                        st.warning("No food items detected")
                col_idx += 1
            
            if show_depth:
                with display_cols[col_idx]:
                    st.markdown("#### Depth Estimation")
                    colored_depth = colorize_depth(depth_map, colormap)
                    st.image(colored_depth, use_column_width=True)
                    st.caption(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        
        # Show analysis results
        if show_analysis and len(masks) > 0:
            st.markdown("### 📊 Depth Analysis per Food Item")
            analysis_results = analyze_depth_per_food_item(
                depth_map, masks, boxes, class_names, labels
            )
            
            if analysis_results:
                import pandas as pd
                
                df_data = []
                for result in analysis_results:
                    df_data.append({
                        'Food Class': result['class_name'].title(),
                        'Mask Area (pixels)': result['mask_area'],
                        'Min Depth': f"{result['depth_min']:.3f}",
                        'Max Depth': f"{result['depth_max']:.3f}",
                        'Mean Depth': f"{result['depth_mean']:.3f}",
                        'Depth Std': f"{result['depth_std']:.3f}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Show statistics
                st.markdown("#### Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Food Items", len(analysis_results))
                
                with col2:
                    total_area = sum(r['mask_area'] for r in analysis_results)
                    st.metric("Total Segmented Area", f"{total_area:,} pixels")
                
                with col3:
                    avg_depth = np.mean([r['depth_mean'] for r in analysis_results])
                    st.metric("Average Depth", f"{avg_depth:.3f}")
            
            else:
                st.info("No valid food items found for depth analysis")
        
        st.markdown("---")


if __name__ == "__main__":
    main()