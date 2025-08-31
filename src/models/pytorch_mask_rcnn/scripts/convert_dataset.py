#!/usr/bin/env python3
"""
Convert RTE8 dataset format to COCO format for PyTorch training
"""

import json
import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image

def convert_rte8_to_coco(data_root):
    """Convert RTE8 dataset to COCO format"""
    
    data_root = Path(data_root)
    print(f"Converting dataset at: {data_root}")
    
    # Define 7-class mapping
    class_mapping = {
        # CARB
        'rice': 'carb', 'bread': 'carb', 'noodles': 'carb', 'pasta': 'carb',
        'corn': 'carb', 'potato': 'carb', 'french fries': 'carb',
        'hamburg': 'carb', 'pizza': 'carb', 'hanamaki baozi': 'carb',
        'wonton dumplings': 'carb', 'pie': 'carb',
        
        # MEAT
        'chicken duck': 'meat', 'pork': 'meat', 'lamb': 'meat', 'steak': 'meat',
        'fried meat': 'meat', 'sausage': 'meat', 'fish': 'meat', 'shrimp': 'meat',
        'crab': 'meat', 'shellfish': 'meat',
        
        # VEGETABLE
        'carrot': 'vegetable', 'broccoli': 'vegetable', 'cabbage': 'vegetable',
        'cauliflower': 'vegetable', 'asparagus': 'vegetable', 'bamboo shoots': 'vegetable',
        'bean sprouts': 'vegetable', 'french beans': 'vegetable', 'green beans': 'vegetable',
        'snow peas': 'vegetable', 'celery stick': 'vegetable', 'cucumber': 'vegetable',
        'eggplant': 'vegetable', 'garlic': 'vegetable', 'ginger': 'vegetable',
        'lettuce': 'vegetable', 'okra': 'vegetable', 'onion': 'vegetable',
        'radish': 'vegetable', 'tomato': 'vegetable', 'mushroom': 'vegetable',
        
        # FRUITS
        'apple': 'fruits', 'banana': 'fruits', 'orange': 'fruits',
        'strawberry': 'fruits', 'grapes': 'fruits', 'watermelon': 'fruits',
        
        # EGG
        'egg': 'egg', 'fried egg': 'egg', 'boiled egg': 'egg',
        
        # GRAVY
        'soup': 'gravy', 'sauce': 'gravy', 'gravy': 'gravy',
        'dressing': 'gravy', 'broth': 'gravy',
        
        # OTHERS
        'cheese': 'others', 'tofu': 'others', 'nuts': 'others',
        'beans': 'others', 'yogurt': 'others'
    }
    
    class_names = ["carb", "meat", "vegetable", "fruits", "egg", "gravy", "others"]
    
    for split in ['train', 'test']:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"Split {split} not found, skipping...")
            continue
            
        img_dir = split_dir / 'img'
        ann_dir = split_dir / 'ann'
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"Missing img or ann directory for {split}, skipping...")
            continue
        
        # Create COCO structure
        output_dir = data_root / split / 'coco'
        output_dir.mkdir(exist_ok=True)
        
        images_out = output_dir / 'images'
        images_out.mkdir(exist_ok=True)
        
        # COCO format data
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': i+1, 'name': name} for i, name in enumerate(class_names)
            ]
        }
        
        image_id = 1
        annotation_id = 1
        
        # Process each image
        for img_file in img_dir.glob('*.jpg'):
            ann_file = ann_dir / f"{img_file.stem}.jpg.json"
            
            if not ann_file.exists():
                continue
            
            # Load image
            image = Image.open(img_file)
            width, height = image.size
            
            # Copy image to output
            output_img_path = images_out / f"{image_id:06d}.jpg"
            image.save(output_img_path)
            
            # Add image info
            coco_data['images'].append({
                'id': image_id,
                'file_name': f"{image_id:06d}.jpg",
                'width': width,
                'height': height
            })
            
            # Load annotations
            try:
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                
                # Process shapes (objects)
                for shape in ann_data.get('shapes', []):
                    label = shape.get('label', '').lower()
                    points = shape.get('points', [])
                    
                    if len(points) < 3:  # Need at least 3 points for polygon
                        continue
                    
                    # Map label to 7-class system
                    mapped_label = class_mapping.get(label, 'others')
                    category_id = class_names.index(mapped_label) + 1
                    
                    # Convert polygon to mask and bbox
                    points_array = np.array(points, dtype=np.int32)
                    
                    # Calculate bounding box
                    x_min = int(np.min(points_array[:, 0]))
                    y_min = int(np.min(points_array[:, 1]))
                    x_max = int(np.max(points_array[:, 0]))
                    y_max = int(np.max(points_array[:, 1]))
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    
                    if bbox_width <= 0 or bbox_height <= 0:
                        continue
                    
                    # Create mask
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [points_array], 1)
                    
                    # Calculate area
                    area = int(np.sum(mask))
                    
                    if area == 0:
                        continue
                    
                    # Convert mask to RLE (simplified - just store polygon)
                    segmentation = [points_array.flatten().tolist()]
                    
                    # Add annotation
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [x_min, y_min, bbox_width, bbox_height],
                        'area': area,
                        'segmentation': segmentation,
                        'iscrowd': 0
                    })
                    
                    annotation_id += 1
                    
            except Exception as e:
                print(f"Error processing {ann_file}: {e}")
                continue
            
            image_id += 1
        
        # Save COCO annotations
        ann_output_dir = output_dir / 'annotations'
        ann_output_dir.mkdir(exist_ok=True)
        
        with open(ann_output_dir / 'instances.json', 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✅ Converted {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    print("🎉 Dataset conversion completed!")
    return str(data_root / 'train' / 'coco'), str(data_root / 'test' / 'coco')

if __name__ == "__main__":
    data_root = "../../data/merged/rte8_7class_food_dataset"
    
    if Path(data_root).exists():
        convert_rte8_to_coco(data_root)
    else:
        print(f"Dataset not found at {data_root}")
        
        # Check for alternative datasets
        alt_paths = [
            "../../data/classification/food_7class_dataset",
            "../../sam2_finetune/data/merged_food_sam2"
        ]
        
        for path in alt_paths:
            if Path(path).exists():
                print(f"Found alternative dataset: {path}")
                break