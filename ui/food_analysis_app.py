#!/usr/bin/env python3
"""
Food Analysis Streamlit App

Simple interface to demonstrate segmentation and depth estimation outputs.
Calls the correct methods from the nutrition pipeline for each model.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import logging

import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

# Add repo src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

try:
    from src.pipeline.nutrition_pipeline import NutritionPipeline, FoodItem
    from src.volume.volume_calculator import VolumeCalculator
except ImportError as e:
    st.error(f"Could not import nutrition pipeline or volume calculator: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Streamlit page
st.set_page_config(
    page_title="Food Analysis Demo",
    page_icon="🍎",
    layout="wide"
)

# Set matplotlib backend
plt.switch_backend('Agg')


class FoodAnalysisApp:
    """Streamlit app for food segmentation and depth analysis."""
    
    def __init__(self):
        self.pipeline = None
        self.volume_calculator = VolumeCalculator()
    
    @st.cache_resource
    def load_pipeline(_self) -> NutritionPipeline:
        """Load and cache the nutrition pipeline."""
        try:
            # Initialize pipeline with available checkpoints
            pipeline = NutritionPipeline(
                depth_encoder="vitb",  # Use medium-size encoder
                enable_reference_scale=False  # Disable for demo
            )
            return pipeline
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            raise
    
    def display_segmentation_results(self, image: np.ndarray, food_items: list, 
                                   show_masks: bool = True) -> np.ndarray:
        """Display segmentation results with bounding boxes and masks."""
        if not food_items:
            st.info("No food items detected")
            return image
        
        # Create visualization
        vis_image = image.copy()
        
        # Color map for different classes
        colors = [
            (255, 0, 0),    # fruit - red
            (0, 255, 0),    # vegetable - green  
            (0, 0, 255),    # carbohydrate - blue
            (255, 255, 0),  # protein - yellow
            (255, 0, 255),  # dairy - magenta
            (0, 255, 255),  # fat - cyan
            (128, 128, 128) # other - gray
        ]
        
        overlay = vis_image.copy()
        
        for i, food_item in enumerate(food_items):
            # Get class-specific color
            class_idx = hash(food_item.broad_class) % len(colors)
            color = colors[class_idx]
            
            # Draw bounding box
            x1, y1, x2, y2 = food_item.bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw mask overlay if requested
            if show_masks and food_item.mask is not None:
                mask = food_item.mask
                if mask.shape[:2] != vis_image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), 
                                    (vis_image.shape[1], vis_image.shape[0]))
                
                # Create colored mask
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                
                # Blend with original image
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            
            # Create label
            label = f"{food_item.broad_class}: {food_item.confidence_broad:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_image, (x1, y1-25), (x1+w, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Combine original with overlay if masks were drawn
        if show_masks:
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        
        return vis_image
    
    def display_depth_results(self, depth_map: np.ndarray) -> Tuple[np.ndarray, plt.Figure]:
        """Display depth estimation results."""
        # Normalize depth map for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Create colored depth map
        depth_colored = plt.cm.viridis(depth_normalized)
        depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Create matplotlib figure for better visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw depth map
        im1 = ax1.imshow(depth_map, cmap='viridis')
        ax1.set_title('Raw Depth Map')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Normalized depth map
        im2 = ax2.imshow(depth_normalized, cmap='plasma')
        ax2.set_title('Normalized Depth Map')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        
        return depth_colored, fig
    
    def create_detection_summary(self, food_items: list) -> None:
        """Create a summary table of detected food items."""
        if not food_items:
            return
        
        st.subheader("🍽️ Detection Summary")
        
        # Create summary data
        summary_data = []
        for item in food_items:
            x1, y1, x2, y2 = item.bbox
            area = (x2 - x1) * (y2 - y1)
            
            summary_data.append({
                "Food Class": item.broad_class.title(),
                "Confidence": f"{item.confidence_broad:.3f}",
                "Bounding Box": f"({x1}, {y1}, {x2}, {y2})",
                "Area (pixels)": f"{area:,}",
                "Mask Available": "✓" if item.mask is not None else "✗"
            })
        
        # Display as table
        import pandas as pd
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
    
    def calculate_volumes(self, food_items: list, depth_map: np.ndarray, method: str = "enhanced") -> list:
        """Calculate volume for each segmented food item."""
        volume_results = []
        
        for i, food_item in enumerate(food_items):
            if food_item.mask is not None:
                try:
                    # Ensure mask dimensions match depth map
                    mask = food_item.mask
                    if mask.shape[:2] != depth_map.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), 
                                        (depth_map.shape[1], depth_map.shape[0]))
                    
                    # Calculate volume using selected method
                    volume_result = self.volume_calculator.calculate_volume(
                        mask, depth_map, method=method
                    )
                    
                    volume_results.append({
                        'food_item': food_item,
                        'item_id': i,
                        'volume_result': volume_result
                    })
                    
                except Exception as e:
                    st.warning(f"Volume calculation failed for item {i}: {e}")
                    continue
            else:
                st.warning(f"No mask available for volume calculation of item {i}")
        
        return volume_results
    
    def display_volume_results(self, volume_results: list) -> None:
        """Display volume calculation results."""
        if not volume_results:
            st.info("No volume calculations available")
            return
        
        st.subheader("📏 Volume Analysis")
        
        # Create summary table
        volume_data = []
        total_volume = 0
        
        for result in volume_results:
            food_item = result['food_item']
            volume_result = result['volume_result']
            
            volume_data.append({
                "Food Item": f"{food_item.broad_class.title()}",
                "Volume (ml)": f"{volume_result.volume_ml:.2f}",
                "Confidence": volume_result.confidence.title(),
                "Mask Area": f"{volume_result.mask_area_pixels:,} px",
                "Avg Depth": f"{volume_result.avg_depth:.4f}",
                "Surface Area": f"{volume_result.surface_area_estimate:.2f} cm²"
            })
            
            total_volume += volume_result.volume_ml
        
        # Display volume table
        import pandas as pd
        df = pd.DataFrame(volume_data)
        st.dataframe(df, use_container_width=True)
        
        # Volume summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Volume", f"{total_volume:.2f} ml")
        
        with col2:
            avg_volume = total_volume / len(volume_results) if volume_results else 0
            st.metric("Average Volume", f"{avg_volume:.2f} ml")
        
        with col3:
            high_conf_count = sum(1 for r in volume_results if r['volume_result'].confidence == 'high')
            st.metric("High Confidence", f"{high_conf_count}/{len(volume_results)}")
        
        with col4:
            max_volume = max((r['volume_result'].volume_ml for r in volume_results), default=0)
            st.metric("Largest Portion", f"{max_volume:.2f} ml")
        
        # Volume breakdown chart
        if len(volume_results) > 1:
            st.markdown("#### Volume Distribution")
            
            # Pie chart of volumes
            labels = [f"{r['food_item'].broad_class.title()}" for r in volume_results]
            volumes = [r['volume_result'].volume_ml for r in volume_results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart
            ax1.pie(volumes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Volume Distribution by Food Class')
            
            # Bar chart
            bars = ax2.bar(labels, volumes)
            ax2.set_title('Volume by Food Item (ml)')
            ax2.set_ylabel('Volume (ml)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Color bars by confidence
            colors = {'high': 'green', 'medium': 'orange', 'low': 'red'}
            for bar, result in zip(bars, volume_results):
                confidence = result['volume_result'].confidence
                bar.set_color(colors.get(confidence, 'blue'))
            
            plt.tight_layout()
            st.pyplot(fig)
            
        # Detailed volume information
        with st.expander("📊 Detailed Volume Information"):
            for result in volume_results:
                food_item = result['food_item']
                volume_result = result['volume_result']
                
                st.markdown(f"**{food_item.broad_class.title()} - Item {result['item_id']}**")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write(f"Volume: {volume_result.volume_ml:.2f} ml")
                    st.write(f"Confidence: {volume_result.confidence}")
                    st.write(f"Mask Area: {volume_result.mask_area_pixels:,} pixels")
                
                with detail_col2:
                    st.write(f"Average Depth: {volume_result.avg_depth:.4f}")
                    st.write(f"Depth Variance: {volume_result.depth_variance:.6f}")
                    st.write(f"Surface Area: {volume_result.surface_area_estimate:.2f} cm²")
                
                st.markdown("---")
    
    def run(self):
        """Main Streamlit app."""
        st.title("🍎 Food Analysis Demo")
        st.markdown("Demonstrates segmentation and depth estimation using the correct pipeline methods")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            confidence_threshold = st.slider(
                "Detection Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
            show_masks = st.checkbox("Show Segmentation Masks", value=True)
            show_depth = st.checkbox("Show Depth Estimation", value=True)
            calculate_volume = st.checkbox("Calculate Volumes", value=True)
            
            if calculate_volume:
                st.markdown("**Volume Settings**")
                volume_method = st.selectbox(
                    "Volume Calculation Method",
                    ["enhanced", "simple", "surface_integration"],
                    index=0,
                    help="Enhanced: Layer-based integration, Simple: Area × depth, Surface: 3D surface calculation"
                )
            else:
                volume_method = "enhanced"
            
            st.markdown("---")
            st.markdown("**Model Status**")
            
            # Show GPU/CPU status
            device_info = "CUDA" if torch.cuda.is_available() else "CPU"
            st.info(f"**Device**: {device_info}")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Food Image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing food items for analysis"
        )
        
        if uploaded_file is None:
            st.info("👆 Upload a food image to begin analysis")
            
            # Show example
            st.markdown("### 📖 About This Demo")
            st.markdown("""
            This app demonstrates the core functionality of the food analysis pipeline:
            
            **🎯 Segmentation Model**: 
            - Uses Mask R-CNN with ResNet backbone
            - Detects 7 food classes: fruit, vegetable, carbohydrate, protein, dairy, fat, other
            - Provides bounding boxes, masks, and confidence scores
            - Method: `pipeline._segment_image(image, conf_threshold)`
            
            **📏 Depth Model**: 
            - Uses Depth Anything for monocular depth estimation
            - Estimates relative depth for each pixel
            - Enables volume calculation when combined with segmentation
            - Method: `pipeline._estimate_depth(image)`
            
            **🔧 Correct Method Calls**:
            - Segmentation: `food_items = pipeline._segment_image(image, conf_threshold)`
            - Depth: `depth_map = pipeline._estimate_depth(image)`
            - Volume: `volume_result = volume_calculator.calculate_volume(mask, depth_map, method)`
            """)
            return
        
        # Load pipeline
        with st.spinner("Loading models..."):
            try:
                self.pipeline = self.load_pipeline()
                st.success("✅ Pipeline loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load pipeline: {e}")
                return
        
        # Process uploaded image
        try:
            # Read image
            file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Could not read the uploaded image")
                return
                
            st.markdown("## 📸 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption=f"Original Image - {image.shape[1]}x{image.shape[0]} pixels",
                    use_column_width=True)
            
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
        
        # Run segmentation analysis
        st.markdown("## 🎯 Segmentation Analysis")
        
        with st.spinner("Running segmentation..."):
            try:
                # UML-COMPLIANT METHOD CALL: MaskRCNNSegmentation.segment_image()
                # This calls the pipeline's internal _segment_image method
                food_items = self.pipeline._segment_image(image, confidence_threshold)
                
                if food_items:
                    st.success(f"✅ **Segmentation Result**: Detected {len(food_items)} food items")
                    
                    # Display UML-compliant SegmentationResult outputs
                    st.markdown("#### 📋 SegmentationResult Output (UML-compliant)")
                    
                    # Create summary showing UML-specified outputs
                    seg_col1, seg_col2 = st.columns(2)
                    
                    with seg_col1:
                        st.markdown("**Detected Objects:**")
                        for i, food_item in enumerate(food_items):
                            with st.expander(f"Food Item {i+1}: {food_item.broad_class.title()}"):
                                st.write(f"**Class**: {food_item.broad_class}")
                                st.write(f"**Confidence**: {food_item.confidence_broad:.3f}")
                                st.write(f"**Bounding Box**: {food_item.bbox}")
                                st.write(f"**Has Mask**: {'✓' if food_item.mask is not None else '✗'}")
                                if hasattr(food_item, 'specific_class') and food_item.specific_class:
                                    st.write(f"**Specific Class**: {food_item.specific_class}")
                    
                    with seg_col2:
                        # Display segmentation visualization
                        seg_viz = self.display_segmentation_results(image, food_items, show_masks)
                        st.image(cv2.cvtColor(seg_viz, cv2.COLOR_BGR2RGB),
                               caption="Segmentation Visualization with Bounding Boxes & Masks",
                               use_column_width=True)
                    
                    # UML SegmentationResult data structure display
                    st.markdown("#### 🔍 UML SegmentationResult Structure")
                    st.markdown("""
                    **SegmentationResult** contains:
                    - `boxes`: Bounding box coordinates for each detection
                    - `masks`: Binary segmentation masks  
                    - `labels`: Class labels (7 food categories)
                    - `scores`: Confidence scores
                    - `get_food_items()`: Converts to FoodItem objects
                    """)
                    
                else:
                    st.warning("No food items detected. Try lowering the confidence threshold.")
                    
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
                st.exception(e)
        
        # Run depth estimation
        if show_depth:
            st.markdown("## 📏 Depth Estimation")
            
            with st.spinner("Estimating depth..."):
                try:
                    # UML-COMPLIANT METHOD CALL: DepthAnythingModel.estimate_depth()
                    # This calls the pipeline's internal _estimate_depth method
                    depth_map = self.pipeline._estimate_depth(image)
                    
                    st.success("✅ **Depth Estimation Complete**")
                    
                    # UML-compliant DepthResult outputs
                    st.markdown("#### 📊 DepthResult Output (UML-compliant)")
                    
                    depth_col1, depth_col2 = st.columns(2)
                    
                    with depth_col1:
                        # Display depth visualization
                        depth_colored, depth_fig = self.display_depth_results(depth_map)
                        st.image(depth_colored,
                               caption="Depth Map Visualization",
                               use_column_width=True)
                    
                    with depth_col2:
                        # UML DepthResult.depth_stats
                        st.markdown("**Depth Statistics (depth_stats):**")
                        depth_stats = {
                            "min_depth": float(depth_map.min()),
                            "max_depth": float(depth_map.max()), 
                            "mean_depth": float(depth_map.mean()),
                            "std_depth": float(depth_map.std()),
                            "depth_range": float(depth_map.max() - depth_map.min())
                        }
                        
                        for key, value in depth_stats.items():
                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                        
                        st.pyplot(depth_fig)
                    
                    # UML DepthResult data structure display
                    st.markdown("#### 🔍 UML DepthResult Structure")
                    st.markdown("""
                    **DepthResult** contains:
                    - `depth_map`: Dense depth values (np.ndarray)
                    - `depth_stats`: Statistical analysis (Dict)
                    - `visualize_depth()`: Creates colored depth visualization
                    """)
                        
                except Exception as e:
                    st.error(f"Depth estimation failed: {e}")
                    st.exception(e)
                    depth_map = None
        
        # Volume Analysis (only if both segmentation and depth are available and enabled)
        if (calculate_volume and 'food_items' in locals() and food_items and 
            'depth_map' in locals() and depth_map is not None):
            st.markdown("## 🧮 Volume Analysis")
            st.markdown("Combining segmentation masks with depth estimation for 3D volume calculation")
            
            with st.spinner("Calculating volumes for each food item..."):
                try:
                    # Calculate volumes for each segmented item
                    volume_results = self.calculate_volumes(food_items, depth_map, volume_method)
                    
                    if volume_results:
                        st.success(f"✅ **Volume Calculation Complete**: Analyzed {len(volume_results)} food items")
                        
                        # Display volume results
                        self.display_volume_results(volume_results)
                        
                        # Volume calculation methodology
                        with st.expander("🔬 Volume Calculation Methodology"):
                            st.markdown("""
                            **Enhanced 3D Volume Estimation**:
                            
                            1. **Mask-Depth Integration**: Combines segmentation masks with depth maps
                            2. **Layer-based Calculation**: Divides food items into depth layers for accurate volume estimation
                            3. **Real-world Scaling**: Converts pixel measurements to milliliters using calibrated scaling factors
                            4. **Confidence Assessment**: Evaluates volume reliability based on:
                               - Mask area size (larger = more reliable)
                               - Depth variance (lower variance = more consistent)
                               - Depth range reasonableness
                               - Volume range validation (10ml - 2L for food portions)
                            
                            **Volume Calculation Methods Available**:
                            - `simple`: Basic area × average depth
                            - `enhanced`: Layer-based depth integration (current method)
                            - `surface_integration`: 3D surface-based calculation
                            """)
                    else:
                        st.warning("No volumes could be calculated. Ensure food items have segmentation masks.")
                        
                except Exception as e:
                    st.error(f"Volume calculation failed: {e}")
                    st.exception(e)
        else:
            if 'food_items' not in locals() or not food_items:
                st.info("Volume analysis requires food items to be detected first.")
            elif 'depth_map' not in locals() or depth_map is None:
                st.info("Volume analysis requires depth estimation to be enabled.")
        
        # Model information
        st.markdown("---")
        st.markdown("### ℹ️ UML Architecture Summary")
        
        arch_col1, arch_col2, arch_col3 = st.columns(3)
        
        with arch_col1:
            st.markdown("**MaskRCNNSegmentation**")
            st.code("""
class MaskRCNNSegmentation:
  + model: torch.nn.Module
  + device: str  
  + class_names: List[str]
  + segment_image(): SegmentationResult
            """)
            st.markdown("**7 Food Classes:**")
            classes = ["fruit", "vegetable", "carbohydrate", "protein", "dairy", "fat", "other"]
            for cls in classes:
                st.write(f"- {cls}")
        
        with arch_col2:
            st.markdown("**DepthAnythingModel**")
            st.code("""
class DepthAnythingModel:
  + model: DepthAnything
  + device: str
  + encoder: str
  + estimate_depth(): np.ndarray
            """)
            st.markdown("**Depth Processing:**")
            st.write("- Monocular depth estimation")
            st.write("- ViT-based encoder")
            st.write("- Dense depth maps")
        
        with arch_col3:
            st.markdown("**Output Classes (UML)**")
            st.code("""
SegmentationResult:
  + boxes: torch.Tensor
  + masks: torch.Tensor  
  + labels: torch.Tensor
  + scores: torch.Tensor

DepthResult:
  + depth_map: np.ndarray
  + depth_stats: Dict
            """)
        
        # Training status info
        st.markdown("### 🔄 Current Training Status")
        st.info("ResNet-152 training is currently running in the background on GPU. Check logs for progress.")


def main():
    """Run the Streamlit app."""
    app = FoodAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()