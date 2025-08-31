#!/usr/bin/env python3
"""
Convert initial dataset to 7-class COCO format for Mask R-CNN training
Based on ResNet-50 backbone requirements
"""

import json
import os
import base64
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import shutil
import io

def decode_bitmap_mask(bitmap_data, origin, img_shape):
    """Decode base64 bitmap mask"""
    try:
        # Decode base64
        binary_data = base64.b64decode(bitmap_data)
        
        # Load as PNG image
        mask_img = Image.open(io.BytesIO(binary_data))
        mask_array = np.array(mask_img)
        
        # Handle different formats
        if len(mask_array.shape) == 3:
            # RGB image - use first channel or convert to grayscale
            if mask_array.shape[2] == 3:
                mask_array = np.dot(mask_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                mask_array = mask_array[:,:,0]
        
        # Create full-size mask
        full_mask = np.zeros(img_shape, dtype=np.uint8)
        
        # Place mask at origin
        x_start, y_start = origin
        y_end = min(y_start + mask_array.shape[0], img_shape[0])
        x_end = min(x_start + mask_array.shape[1], img_shape[1])
        
        # Ensure we don't go out of bounds
        mask_h = y_end - y_start
        mask_w = x_end - x_start
        
        if mask_h > 0 and mask_w > 0:
            full_mask[y_start:y_end, x_start:x_end] = mask_array[:mask_h, :mask_w] > 0
        
        return full_mask
        
    except Exception as e:
        print(f"Error decoding bitmap: {e}")
        return None

def convert_to_7class_coco(data_root, output_dir):
    """Convert initial dataset to 7-class COCO format"""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 7-class mapping based on semantic categories
    class_mapping = {
        # CARB (class 1)
        'rice': 'carb', 'bread': 'carb', 'noodles': 'carb', 'pasta': 'carb',
        'corn': 'carb', 'potato': 'carb', 'french fries': 'carb',
        'hamburg': 'carb', 'pizza': 'carb', 'pie': 'carb', 'dumplings': 'carb',
        
        # MEAT (class 2)  
        'chicken': 'meat', 'duck': 'meat', 'pork': 'meat', 'beef': 'meat',
        'lamb': 'meat', 'steak': 'meat', 'sausage': 'meat', 'fish': 'meat', 
        'shrimp': 'meat', 'crab': 'meat', 'shellfish': 'meat', 'seafood': 'meat',
        
        # VEGETABLE (class 3)
        'carrot': 'vegetable', 'broccoli': 'vegetable', 'cabbage': 'vegetable',
        'cauliflower': 'vegetable', 'asparagus': 'vegetable', 'bamboo shoots': 'vegetable',
        'bean sprouts': 'vegetable', 'green beans': 'vegetable', 'peas': 'vegetable',
        'celery': 'vegetable', 'cucumber': 'vegetable', 'eggplant': 'vegetable',
        'garlic': 'vegetable', 'ginger': 'vegetable', 'lettuce': 'vegetable',
        'onion': 'vegetable', 'radish': 'vegetable', 'tomato': 'vegetable',
        'mushroom': 'vegetable', 'pepper': 'vegetable', 'spinach': 'vegetable',
        
        # FRUITS (class 4)
        'apple': 'fruits', 'banana': 'fruits', 'orange': 'fruits',
        'strawberry': 'fruits', 'grapes': 'fruits', 'watermelon': 'fruits',
        'berries': 'fruits', 'citrus': 'fruits', 'melon': 'fruits',
        
        # EGG (class 5)
        'egg': 'egg', 'eggs': 'egg', 'fried egg': 'egg', 'boiled egg': 'egg',
        
        # GRAVY (class 6) 
        'sauce': 'gravy', 'soup': 'gravy', 'gravy': 'gravy', 'broth': 'gravy',
        'dressing': 'gravy', 'condiment': 'gravy', 'liquid': 'gravy',
        
        # OTHERS (class 7)
        'cheese': 'others', 'tofu': 'others', 'nuts': 'others', 'beans': 'others',
        'dairy': 'others', 'oil': 'others', 'spice': 'others', 'seasoning': 'others'
    }
    
    class_names = ['carb', 'meat', 'vegetable', 'fruits', 'egg', 'gravy', 'others']
    
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
        
        # Create output structure
        out_split_dir = output_dir / split
        out_split_dir.mkdir(exist_ok=True)
        
        out_img_dir = out_split_dir / 'images'
        out_img_dir.mkdir(exist_ok=True)
        
        out_ann_dir = out_split_dir / 'annotations'
        out_ann_dir.mkdir(exist_ok=True)
        
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
        processed_count = 0
        
        print(f"Processing {split} split...")
        
        # Process each image
        for img_file in sorted(img_dir.glob('*.jpg')):
            ann_file = ann_dir / f"{img_file.name}.json"
            
            if not ann_file.exists():
                continue
            
            try:
                # Load image
                image = Image.open(img_file).convert('RGB')
                width, height = image.size
                
                # Copy image to output with sequential naming
                output_img_path = out_img_dir / f"{image_id:06d}.jpg"
                image.save(output_img_path)
                
                # Add image info
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': f"{image_id:06d}.jpg",
                    'width': width,
                    'height': height,
                    'original_name': img_file.name
                })
                
                # Load annotations
                with open(ann_file, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                
                # Process objects
                for obj in ann_data.get('objects', []):
                    class_title = obj.get('classTitle', '').lower()
                    
                    # Map to 7-class system
                    mapped_class = None
                    for original, mapped in class_mapping.items():
                        if original in class_title:
                            mapped_class = mapped
                            break
                    
                    if mapped_class is None:
                        # Default to 'others' if not found
                        mapped_class = 'others'
                    
                    category_id = class_names.index(mapped_class) + 1
                    
                    # Handle bitmap geometry
                    if obj.get('geometryType') == 'bitmap' and 'bitmap' in obj:
                        bitmap = obj['bitmap']
                        bitmap_data = bitmap['data']
                        origin = bitmap['origin']  # [x, y]
                        
                        # Decode bitmap mask
                        import io
                        mask = decode_bitmap_mask(bitmap_data, origin, (height, width))
                        
                        if mask is None or np.sum(mask) == 0:
                            continue
                        
                        # Calculate bounding box
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        
                        if not np.any(rows) or not np.any(cols):
                            continue
                            
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        
                        bbox_width = int(x_max - x_min + 1)
                        bbox_height = int(y_max - y_min + 1)
                        
                        if bbox_width <= 0 or bbox_height <= 0:
                            continue
                        
                        # Calculate area
                        area = int(np.sum(mask))
                        
                        # Create RLE segmentation (simplified)
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        segmentation = []
                        for contour in contours:
                            if len(contour) >= 3:
                                seg = contour.flatten().tolist()
                                if len(seg) >= 6:  # At least 3 points
                                    segmentation.append(seg)
                        
                        if not segmentation:
                            continue
                        
                        # Add annotation
                        coco_data['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': [int(x_min), int(y_min), bbox_width, bbox_height],
                            'area': area,
                            'segmentation': segmentation,
                            'iscrowd': 0,
                            'original_class': class_title
                        })
                        
                        annotation_id += 1
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} images...")
                
                image_id += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        # Save COCO annotations
        with open(out_ann_dir / 'instances.json', 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Completed {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        
        # Print class distribution
        class_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            class_name = class_names[cat_id - 1]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Class distribution for {split}:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    
    print("Dataset conversion completed!")
    return output_dir

if __name__ == "__main__":
    data_root = "../../data/initial"
    output_dir = "../data/7class_maskrcnn_dataset" 
    
    if Path(data_root).exists():
        result_dir = convert_to_7class_coco(data_root, output_dir)
        print(f"Dataset saved to: {result_dir}")
    else:
        print(f"Dataset not found at {data_root}")