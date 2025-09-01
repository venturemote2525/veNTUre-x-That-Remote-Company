#!/usr/bin/env python3
"""
Convert YOLO format dataset to COCO format for Mask R-CNN training
"""

import json
import os
from pathlib import Path
from PIL import Image
import yaml
import cv2
import numpy as np

def yolo_to_coco(yolo_dataset_path, output_path):
    """Convert YOLO format to COCO format"""
    
    yolo_path = Path(yolo_dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO data.yaml
    data_yaml = yolo_path / 'data.yaml'
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    print(f"Converting dataset with classes: {class_names}")
    
    for split in ['train', 'val']:
        split_path = yolo_path / split
        if not split_path.exists():
            print(f"Split {split} not found, skipping...")
            continue
        
        # Create output structure
        output_split = output_path / split
        output_split.mkdir(exist_ok=True)
        
        output_images = output_split / 'images'
        output_images.mkdir(exist_ok=True)
        
        output_annotations = output_split / 'annotations'
        output_annotations.mkdir(exist_ok=True)
        
        # COCO annotation structure
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': i+1, 'name': name, 'supercategory': 'food'}
                for i, name in enumerate(class_names)
            ]
        }
        
        image_id = 1
        annotation_id = 1
        
        # Process images
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        if not images_path.exists():
            print(f"Images path {images_path} not found, skipping...")
            continue
        
        for img_file in sorted(images_path.glob('*.jpg')):
            try:
                # Load and copy image
                image = Image.open(img_file).convert('RGB')
                width, height = image.size
                
                # Copy image to output
                output_img_path = output_images / f"{image_id:06d}.jpg"
                image.save(output_img_path)
                
                # Add image info
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': f"{image_id:06d}.jpg",
                    'width': width,
                    'height': height
                })
                
                # Process labels if they exist
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1]) * width
                                y_center = float(parts[2]) * height
                                bbox_width = float(parts[3]) * width
                                bbox_height = float(parts[4]) * height
                                
                                # Convert to COCO bbox format (x, y, width, height)
                                x1 = x_center - bbox_width / 2
                                y1 = y_center - bbox_height / 2
                                
                                # Clamp coordinates to image bounds and ensure positive dimensions
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(width, x1 + bbox_width)
                                y2 = min(height, y1 + bbox_height)
                                
                                # Skip invalid boxes (too small or zero area)
                                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                                    continue
                                
                                # Update bbox dimensions after clamping
                                bbox_width = x2 - x1
                                bbox_height = y2 - y1
                                
                                # Create simple rectangular mask from bbox
                                mask = np.zeros((height, width), dtype=np.uint8)
                                x1_int = int(x1)
                                y1_int = int(y1)
                                x2_int = int(x2)
                                y2_int = int(y2)
                                
                                mask[y1_int:y2_int, x1_int:x2_int] = 1
                                
                                # Create segmentation from mask (simple rectangle)
                                segmentation = [[x1_int, y1_int, x2_int, y1_int, x2_int, y2_int, x1_int, y2_int]]
                                
                                # Add annotation
                                coco_data['annotations'].append({
                                    'id': annotation_id,
                                    'image_id': image_id,
                                    'category_id': class_id + 1,  # COCO categories are 1-indexed
                                    'bbox': [x1, y1, bbox_width, bbox_height],
                                    'area': bbox_width * bbox_height,
                                    'segmentation': segmentation,
                                    'iscrowd': 0
                                })
                                
                                annotation_id += 1
                
                image_id += 1
                
                if image_id % 100 == 0:
                    print(f"Processed {image_id} images...")
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        # Save COCO annotations
        with open(output_annotations / 'instances.json', 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Converted {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    return output_path

if __name__ == "__main__":
    yolo_dataset = "../data/merged/merged_5class_food_dataset"
    output_dataset = "../data/coco_5class_food_dataset"
    
    if Path(yolo_dataset).exists():
        result_path = yolo_to_coco(yolo_dataset, output_dataset)
        print(f"COCO dataset created at: {result_path}")
        
        # Update paths in start_training.py to use the new dataset
        print(f"Update your training script to use: {result_path}")
    else:
        print(f"YOLO dataset not found at {yolo_dataset}")