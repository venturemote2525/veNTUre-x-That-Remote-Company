#!/usr/bin/env python3
"""
Streamlit Nutrition Analysis App

Unified interface combining food segmentation, depth estimation, and nutritional analysis.
Replaces separate segmentation_app.py and depth_anything_app.py with integrated functionality.

Launch:
  python -m streamlit run src/pipeline/streamlit_app.py
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging

import numpy as np
import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Add repo src to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

try:
    from src.pipeline.nutrition_pipeline import NutritionPipeline, NutritionAnalysisResult
    from src.specific_classification.interface import create_specific_classifier
    from src.nutrition.database import NutritionDatabase
    from src.volume.volume_calculator import VolumeCalculator
except ImportError as e:
    st.error(f"Could not import nutrition analysis modules: {e}")
    st.stop()


# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure page
st.set_page_config(
    page_title="Food Nutrition Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib backend for streamlit
plt.switch_backend('Agg')


@st.cache_resource(show_spinner=False)
def load_nutrition_pipeline(segmentation_checkpoint: Optional[str] = None,
                           depth_encoder: str = "vitb",
                           use_specific_classifier: bool = False,
                           specific_classifier_type: str = "mock") -> NutritionPipeline:
    """Load and cache the nutrition analysis pipeline."""
    try:
        # Create specific classifier if requested
        specific_classifier = None
        if use_specific_classifier:
            specific_classifier = create_specific_classifier(specific_classifier_type)
        
        pipeline = NutritionPipeline(
            segmentation_checkpoint=segmentation_checkpoint,
            depth_encoder=depth_encoder,
            specific_classifier=specific_classifier
        )
        return pipeline
    except Exception as e:
        st.error(f"Failed to load nutrition pipeline: {e}")
        raise


def display_nutrition_summary(result: NutritionAnalysisResult):
    """Display nutrition summary in a nice format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Calories",
            value=f"{result.total_calories:.1f}",
            help="Total estimated calories for all detected food items"
        )
    
    with col2:
        st.metric(
            label="Total Volume",
            value=f"{result.total_volume_ml:.1f} ml",
            help="Total estimated volume of all food items"
        )
    
    with col3:
        st.metric(
            label="Protein",
            value=f"{result.total_protein_g:.1f}g",
            help="Total estimated protein content"
        )
    
    with col4:
        st.metric(
            label="Food Items",
            value=str(len(result.food_items)),
            help="Number of detected food items"
        )
    
    # Detailed macros
    st.markdown("### 📊 Macronutrient Breakdown")
    macro_col1, macro_col2, macro_col3 = st.columns(3)
    
    with macro_col1:
        st.metric("Carbohydrates", f"{result.total_carbs_g:.1f}g")
    
    with macro_col2:
        st.metric("Fat", f"{result.total_fat_g:.1f}g")
    
    with macro_col3:
        st.metric("Fiber", f"{result.total_fiber_g:.1f}g")


def display_food_items_table(result: NutritionAnalysisResult):
    """Display detailed food items analysis in a table."""
    if not result.food_items:
        st.info("No food items detected")
        return
    
    st.markdown("### 🍽️ Individual Food Items")
    
    # Create DataFrame for display
    data = []
    for item in result.food_items:
        food_item = item.food_item
        row = {
            'Food Class': food_item.broad_class.title(),
            'Specific Item': food_item.specific_class.title() if food_item.specific_class else "N/A",
            'Confidence': f"{food_item.confidence_broad:.3f}",
            'Volume (ml)': f"{item.volume.volume_ml:.1f}",
            'Weight (g)': f"{item.estimated_weight_g:.1f}",
            'Calories': f"{item.total_calories:.1f}",
            'Protein (g)': f"{item.total_protein_g:.1f}",
            'Carbs (g)': f"{item.total_carbs_g:.1f}",
            'Fat (g)': f"{item.total_fat_g:.1f}",
            'Volume Confidence': item.volume.confidence.title()
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def create_nutrition_charts(result: NutritionAnalysisResult):
    """Create visualization charts for nutrition analysis."""
    if not result.food_items:
        return
    
    st.markdown("### 📈 Nutrition Visualizations")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Calories by food item
        fig, ax = plt.subplots(figsize=(8, 6))
        
        food_names = []
        calories = []
        for item in result.food_items:
            name = item.food_item.specific_class or item.food_item.broad_class
            food_names.append(name.title())
            calories.append(item.total_calories)
        
        bars = ax.bar(food_names, calories, color='skyblue')
        ax.set_title('Calories by Food Item')
        ax.set_ylabel('Calories')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, cal in zip(bars, calories):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{cal:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with chart_col2:
        # Macronutrient distribution pie chart
        macros = {
            'Protein': result.total_protein_g * 4,  # 4 cal/g
            'Carbs': result.total_carbs_g * 4,      # 4 cal/g
            'Fat': result.total_fat_g * 9           # 9 cal/g
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = list(macros.values())
        labels = [f'{k}\n{v:.1f} cal' for k, v in macros.items()]
        colors = ['lightcoral', 'lightskyblue', 'lightgreen']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
            ax.set_title('Macronutrient Calorie Distribution')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No nutrition data', ha='center', va='center')
        
        st.pyplot(fig)


def display_image_analysis(image: np.ndarray, result: NutritionAnalysisResult, 
                          show_segmentation: bool = True, show_depth: bool = True):
    """Display original image with analysis overlays."""
    st.markdown("### 🖼️ Image Analysis")
    
    if show_segmentation or show_depth:
        display_cols = st.columns(2 if show_depth else 1)
        col_idx = 0
        
        if show_segmentation:
            with display_cols[col_idx]:
                st.markdown("#### Segmentation Results")
                
                # Create visualization with bounding boxes and labels
                vis_image = image.copy()
                
                for item in result.food_items:
                    food_item = item.food_item
                    x1, y1, x2, y2 = food_item.bbox
                    
                    # Draw bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Create label
                    label_text = food_item.broad_class
                    if food_item.specific_class:
                        label_text = food_item.specific_class
                    
                    label = f"{label_text}: {food_item.confidence_broad:.2f}"
                    
                    # Draw label background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(vis_image, (x1, y1-20), (x1+w, y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(vis_image, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), 
                        caption="Detected Food Items", use_column_width=True)
            
            col_idx += 1
        
        if show_depth and len(display_cols) > col_idx:
            with display_cols[col_idx]:
                st.markdown("#### Depth Estimation")
                st.info("Depth visualization requires re-running depth estimation")
    
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                caption="Original Image", use_column_width=True)


def export_results(result: NutritionAnalysisResult, format_type: str = "json"):
    """Create downloadable export of analysis results."""
    if format_type == "json":
        export_data = result.export_json()
        filename = f"nutrition_analysis_{Path(result.image_path).stem}.json"
        mime = "application/json"
    
    elif format_type == "csv":
        # Create CSV from food items
        data = []
        for item in result.food_items:
            food_item = item.food_item
            row = {
                'food_class': food_item.broad_class,
                'specific_class': food_item.specific_class or '',
                'confidence': food_item.confidence_broad,
                'volume_ml': item.volume.volume_ml,
                'weight_g': item.estimated_weight_g,
                'calories': item.total_calories,
                'protein_g': item.total_protein_g,
                'carbs_g': item.total_carbs_g,
                'fat_g': item.total_fat_g,
                'fiber_g': item.total_fiber_g
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        export_data = df.to_csv(index=False)
        filename = f"nutrition_analysis_{Path(result.image_path).stem}.csv"
        mime = "text/csv"
    
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
    
    return export_data, filename, mime


def main():
    """Main Streamlit application."""
    st.title("🍎 Food Nutrition Analysis")
    st.markdown("Integrated food segmentation, depth estimation, and nutritional analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        depth_encoder = st.selectbox(
            "Depth Encoder",
            options=["vits", "vitb", "vitl"],
            index=1,
            help="Depth Anything encoder size (larger = more accurate, slower)"
        )
        
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for food detection"
        )
        
        # Specific classifier settings
        st.subheader("Specific Classification")
        use_specific_classifier = st.checkbox(
            "Enable Specific Classification",
            value=False,
            help="Use friend's specific food classifier (when available)"
        )
        
        specific_classifier_type = "mock"
        if use_specific_classifier:
            specific_classifier_type = st.selectbox(
                "Classifier Type",
                options=["mock", "friends"],
                index=0,
                help="Type of specific classifier to use"
            )
        
        # Display options
        st.subheader("Display Options")
        show_segmentation = st.checkbox("Show Segmentation", value=True)
        show_detailed_table = st.checkbox("Show Detailed Table", value=True)
        show_charts = st.checkbox("Show Nutrition Charts", value=True)
        
        # Export options
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format", ["json", "csv"])
    
    # Main content area
    uploaded_files = st.file_uploader(
        "Upload Food Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more images of food for nutrition analysis"
    )
    
    if not uploaded_files:
        st.info("👆 Upload food images to begin nutrition analysis")
        
        # Show sample information
        st.markdown("### 🍽️ About This App")
        st.markdown("""
        This integrated nutrition analysis tool combines:
        
        - **🎯 Food Segmentation**: Detects and segments food items using Mask R-CNN
        - **📏 Depth Estimation**: Estimates 3D depth using Depth Anything model
        - **🧮 Volume Calculation**: Calculates food volumes from depth + segmentation
        - **🥗 Nutrition Analysis**: Estimates calories and macronutrients
        - **🎯 Specific Classification**: Optional integration with specific food classifier
        
        **Supported Food Categories:**
        - Fruits, Vegetables, Carbohydrates, Proteins, Dairy, Fats, Other processed foods
        
        **Nutritional Information:**
        - Calories, Protein, Carbohydrates, Fat, Fiber
        - Volume and weight estimates
        - Per-item and total analysis
        """)
        return
    
    # Load pipeline
    with st.spinner("Loading nutrition analysis pipeline..."):
        try:
            pipeline = load_nutrition_pipeline(
                depth_encoder=depth_encoder,
                use_specific_classifier=use_specific_classifier,
                specific_classifier_type=specific_classifier_type
            )
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            return
    
    st.success("✅ Pipeline loaded successfully!")
    
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
        
        # Save temporary file for pipeline processing
        temp_path = f"/tmp/{uploaded_file.name}"
        cv2.imwrite(temp_path, image)
        
        # Run analysis
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            try:
                result = pipeline.analyze_food_image(
                    temp_path,
                    conf_threshold=confidence_threshold
                )
            except Exception as e:
                st.error(f"Analysis failed for {uploaded_file.name}: {e}")
                continue
        
        # Display results
        if not result.food_items:
            st.warning("No food items detected in this image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption="Original Image", use_column_width=True)
            continue
        
        # Nutrition summary
        display_nutrition_summary(result)
        
        # Image analysis
        display_image_analysis(image, result, show_segmentation, False)
        
        # Detailed table
        if show_detailed_table:
            display_food_items_table(result)
        
        # Charts
        if show_charts:
            create_nutrition_charts(result)
        
        # Export functionality
        st.markdown("### 💾 Export Results")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            export_data, filename, mime = export_results(result, export_format)
            st.download_button(
                label=f"Download {export_format.upper()} Report",
                data=export_data,
                file_name=filename,
                mime=mime
            )
        
        with export_col2:
            st.json({
                "summary": {
                    "total_calories": round(result.total_calories, 1),
                    "total_items": len(result.food_items),
                    "analysis_timestamp": result.analysis_timestamp
                }
            })
        
        st.markdown("---")
    
    # Footer
    st.markdown("### 📚 Pipeline Information")
    if 'pipeline' in locals():
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.info(f"**Depth Encoder**: {depth_encoder}")
        
        with info_col2:
            specific_status = "Enabled" if use_specific_classifier else "Disabled"
            st.info(f"**Specific Classification**: {specific_status}")
        
        with info_col3:
            st.info(f"**Detection Confidence**: {confidence_threshold:.2f}")


if __name__ == "__main__":
    main()
