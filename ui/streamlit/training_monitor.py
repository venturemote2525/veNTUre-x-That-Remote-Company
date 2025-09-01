#!/usr/bin/env python3
"""
Streamlit Training Monitor for Food Portion Size Classifier
Real-time monitoring of training progress and GPU metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
import os
import time
from datetime import datetime
from pathlib import Path

# Page config
st.set_page_config(
    page_title="RTX 3060 Training Monitor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_log_file(log_path):
    """Parse training log file for metrics"""
    if not os.path.exists(log_path):
        return None, None, None
    
    epochs = []
    train_losses = []
    val_losses = []
    val_maps = []
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Parse training losses
        train_pattern = r'Epoch (\d+)/\d+.*?Average Training Loss: ([\d.]+)'
        train_matches = re.findall(train_pattern, content)
        
        # Parse validation losses and mAP
        val_pattern = r'Epoch (\d+) Validation - Loss: ([\d.]+), mAP: ([\d.]+)'
        val_matches = re.findall(val_pattern, content)
        
        for epoch, loss in train_matches:
            epochs.append(int(epoch))
            train_losses.append(float(loss))
            
        val_data = {}
        for epoch, val_loss, val_map in val_matches:
            val_data[int(epoch)] = (float(val_loss), float(val_map))
            
        # Align validation data with training epochs
        for epoch in epochs:
            if epoch in val_data:
                val_losses.append(val_data[epoch][0])
                val_maps.append(val_data[epoch][1])
            else:
                val_losses.append(None)
                val_maps.append(None)
                
        return epochs, train_losses, val_losses, val_maps
    except Exception as e:
        st.error(f"Error parsing log file: {e}")
        return None, None, None, None

def get_latest_log_file():
    """Find the most recent training log file"""
    log_dirs = [
        "src/training/logs/swiss_7class_resnet152",
        "src/training/logs/swiss_7class_resnet50", 
        "logs/swiss_7class_resnet152",
        "logs/swiss_7class_resnet50",
        "logs/rtx3060_segmentation"
    ]
    
    latest_file = None
    latest_time = 0
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.startswith('training_') and file.endswith('.log'):
                    file_path = os.path.join(log_dir, file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
                        
    return latest_file

def get_training_config():
    """Extract training configuration from latest log"""
    log_file = get_latest_log_file()
    if not log_file:
        return {}
    
    config = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract config info
        config_patterns = {
            'backbone': r'Backbone: (\w+)',
            'batch_size': r'Batch Size: (\d+)',
            'workers': r'Workers: (\d+)',
            'epochs': r'Total Epochs: (\d+)',
            'classes': r'Total classes.*?(\d+)',
            'train_samples': r'Training samples: (\d+)',
            'val_samples': r'Validation samples: (\d+)'
        }
        
        for key, pattern in config_patterns.items():
            match = re.search(pattern, content)
            if match:
                config[key] = match.group(1)
                
    except Exception as e:
        st.error(f"Error reading config: {e}")
        
    return config

def main():
    st.title("🚀 RTX 3060 Training Monitor")
    st.markdown("### Real-time Food Segmentation Training Dashboard")
    
    # Sidebar
    st.sidebar.header("📊 Training Status")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Get latest log file
    log_file = get_latest_log_file()
    
    if not log_file:
        st.error("❌ No training log files found!")
        st.info("Make sure training is running and log files exist in the expected directories.")
        return
    
    st.sidebar.success(f"📁 Log: {os.path.basename(log_file)}")
    st.sidebar.info(f"📍 Path: {log_file}")
    
    # Parse training data
    epochs, train_losses, val_losses, val_maps = parse_log_file(log_file)
    
    if not epochs:
        st.warning("⚠️ No training data found in log file")
        return
    
    # Training configuration
    config = get_training_config()
    
    # Display config in sidebar
    st.sidebar.header("⚙️ Configuration")
    for key, value in config.items():
        st.sidebar.metric(key.replace('_', ' ').title(), value)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_epoch = max(epochs) if epochs else 0
        total_epochs = int(config.get('epochs', 20))
        st.metric("Current Epoch", f"{current_epoch}/{total_epochs}")
    
    with col2:
        latest_train_loss = train_losses[-1] if train_losses else 0
        st.metric("Latest Train Loss", f"{latest_train_loss:.4f}")
    
    with col3:
        latest_val_loss = [x for x in val_losses if x is not None][-1] if any(x is not None for x in val_losses) else 0
        st.metric("Latest Val Loss", f"{latest_val_loss:.4f}")
    
    with col4:
        latest_map = [x for x in val_maps if x is not None][-1] if any(x is not None for x in val_maps) else 0
        st.metric("Latest mAP", f"{latest_map:.4f}")
    
    # Progress bar
    progress = current_epoch / total_epochs if total_epochs > 0 else 0
    st.progress(progress)
    st.caption(f"Training Progress: {progress*100:.1f}%")
    
    # Loss plots
    st.header("📈 Training Progress")
    
    if len(epochs) > 1:
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training & Validation Loss', 'Training Loss Detail', 'Validation mAP', 'Loss Comparison'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Training and validation loss
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name="Train Loss", line=dict(color="blue")),
            row=1, col=1
        )
        
        val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
        val_loss_clean = [v for v in val_losses if v is not None]
        
        if val_loss_clean:
            fig.add_trace(
                go.Scatter(x=val_epochs, y=val_loss_clean, name="Val Loss", line=dict(color="red")),
                row=1, col=1
            )
        
        # Training loss detail
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name="Train Loss Detail", 
                      mode="lines+markers", line=dict(color="darkblue")),
            row=1, col=2
        )
        
        # Validation mAP
        val_map_clean = [v for v in val_maps if v is not None]
        if val_map_clean:
            fig.add_trace(
                go.Scatter(x=val_epochs, y=val_map_clean, name="mAP", 
                          mode="lines+markers", line=dict(color="green")),
                row=2, col=1
            )
        
        # Loss comparison (last 5 epochs)
        if len(epochs) >= 5:
            recent_epochs = epochs[-5:]
            recent_train = train_losses[-5:]
            
            fig.add_trace(
                go.Bar(x=[f"Epoch {e}" for e in recent_epochs], y=recent_train, 
                      name="Recent Train Loss", marker_color="lightblue"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Training Metrics Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw log display
    st.header("📋 Recent Log Entries")
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Show last 20 lines
        log_lines = log_content.split('\n')[-20:]
        log_text = '\n'.join(log_lines)
        st.code(log_text, language="text")
        
    except Exception as e:
        st.error(f"Error reading log file: {e}")
    
    # File stats
    st.header("📁 File Information")
    try:
        file_stats = os.stat(log_file)
        file_size = file_stats.st_size
        file_modified = datetime.fromtimestamp(file_stats.st_mtime)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Size", f"{file_size:,} bytes")
        with col2:
            st.metric("Last Modified", file_modified.strftime("%Y-%m-%d %H:%M:%S"))
            
    except Exception as e:
        st.error(f"Error getting file stats: {e}")

if __name__ == "__main__":
    main()