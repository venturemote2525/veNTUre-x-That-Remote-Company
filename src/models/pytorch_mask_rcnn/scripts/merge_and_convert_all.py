#!/usr/bin/env python3
"""
Merge all three datasets from initial and convert to 7-class COCO format
- RTE8 train/test data  
- Swiss validation data
- Food dataset from roboflow
Groups all food classes into 7 semantic categories: carb, meat, vegetable, fruits, egg, gravy, others
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
from collections import defaultdict

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

def get_7class_mapping():
    """Comprehensive 7-class mapping for all food types"""
    return {
        # CARB (class 1) - Carbohydrates and starches
        'rice': 'carb', 'bread': 'carb', 'noodles': 'carb', 'pasta': 'carb',
        'corn': 'carb', 'potato': 'carb', 'potatoes': 'carb', 'french fries': 'carb',
        'fries': 'carb', 'hamburg': 'carb', 'hamburger': 'carb', 'pizza': 'carb',
        'pie': 'carb', 'dumplings': 'carb', 'wonton': 'carb', 'baozi': 'carb',
        'hanamaki': 'carb', 'cereal': 'carb', 'oats': 'carb', 'quinoa': 'carb',
        'barley': 'carb', 'wheat': 'carb', 'flour': 'carb', 'tortilla': 'carb',
        'crackers': 'carb', 'chips': 'carb', 'biscuit': 'carb', 'cake': 'carb',
        'muffin': 'carb', 'pancake': 'carb', 'waffle': 'carb',
        
        # MEAT (class 2) - All proteins from animals including seafood
        'chicken': 'meat', 'duck': 'meat', 'pork': 'meat', 'beef': 'meat',
        'lamb': 'meat', 'mutton': 'meat', 'steak': 'meat', 'sausage': 'meat',
        'bacon': 'meat', 'ham': 'meat', 'fish': 'meat', 'salmon': 'meat',
        'tuna': 'meat', 'shrimp': 'meat', 'prawn': 'meat', 'crab': 'meat',
        'lobster': 'meat', 'shellfish': 'meat', 'seafood': 'meat', 'turkey': 'meat',
        'goose': 'meat', 'rabbit': 'meat', 'venison': 'meat', 'fried meat': 'meat',
        'grilled meat': 'meat', 'roast': 'meat', 'meatball': 'meat', 'patty': 'meat',
        'chicken duck': 'meat', 'poultry': 'meat',
        
        # VEGETABLE (class 3) - All vegetables, herbs, and mushrooms
        'carrot': 'vegetable', 'broccoli': 'vegetable', 'cabbage': 'vegetable',
        'cauliflower': 'vegetable', 'asparagus': 'vegetable', 'bamboo shoots': 'vegetable',
        'bean sprouts': 'vegetable', 'green beans': 'vegetable', 'french beans': 'vegetable',
        'peas': 'vegetable', 'snow peas': 'vegetable', 'celery': 'vegetable',
        'cucumber': 'vegetable', 'eggplant': 'vegetable', 'aubergine': 'vegetable',
        'garlic': 'vegetable', 'ginger': 'vegetable', 'lettuce': 'vegetable',
        'onion': 'vegetable', 'radish': 'vegetable', 'tomato': 'vegetable',
        'mushroom': 'vegetable', 'pepper': 'vegetable', 'spinach': 'vegetable',
        'kale': 'vegetable', 'swiss chard': 'vegetable', 'bok choy': 'vegetable',
        'okra': 'vegetable', 'zucchini': 'vegetable', 'squash': 'vegetable',
        'pumpkin': 'vegetable', 'beets': 'vegetable', 'turnip': 'vegetable',
        'leek': 'vegetable', 'scallion': 'vegetable', 'herbs': 'vegetable',
        'parsley': 'vegetable', 'cilantro': 'vegetable', 'basil': 'vegetable',
        
        # FRUITS (class 4) - All fruits and berries
        'apple': 'fruits', 'banana': 'fruits', 'orange': 'fruits', 'citrus': 'fruits',
        'strawberry': 'fruits', 'berries': 'fruits', 'blueberry': 'fruits',
        'raspberry': 'fruits', 'blackberry': 'fruits', 'grapes': 'fruits',
        'watermelon': 'fruits', 'melon': 'fruits', 'cantaloupe': 'fruits',
        'pineapple': 'fruits', 'mango': 'fruits', 'papaya': 'fruits',
        'kiwi': 'fruits', 'pear': 'fruits', 'peach': 'fruits', 'plum': 'fruits',
        'cherry': 'fruits', 'apricot': 'fruits', 'lemon': 'fruits', 'lime': 'fruits',
        'grapefruit': 'fruits', 'coconut': 'fruits', 'avocado': 'fruits',
        
        # EGG (class 5) - All egg preparations
        'egg': 'egg', 'eggs': 'egg', 'fried egg': 'egg', 'boiled egg': 'egg',
        'scrambled egg': 'egg', 'poached egg': 'egg', 'egg white': 'egg',
        'egg yolk': 'egg', 'omelet': 'egg', 'omelette': 'egg',
        
        # GRAVY (class 6) - All liquids, sauces, and condiments
        'sauce': 'gravy', 'soup': 'gravy', 'gravy': 'gravy', 'broth': 'gravy',
        'stock': 'gravy', 'dressing': 'gravy', 'condiment': 'gravy',
        'liquid': 'gravy', 'syrup': 'gravy', 'honey': 'gravy', 'jam': 'gravy',
        'jelly': 'gravy', 'ketchup': 'gravy', 'mustard': 'gravy',
        'mayonnaise': 'gravy', 'oil': 'gravy', 'vinegar': 'gravy',
        'marinade': 'gravy', 'curry': 'gravy', 'paste': 'gravy',
        
        # OTHERS (class 7) - Dairy, nuts, legumes, and miscellaneous
        'cheese': 'others', 'milk': 'others', 'yogurt': 'others', 'butter': 'others',
        'cream': 'others', 'tofu': 'others', 'tempeh': 'others', 'nuts': 'others',
        'almonds': 'others', 'walnuts': 'others', 'peanuts': 'others',
        'beans': 'others', 'lentils': 'others', 'chickpeas': 'others',
        'quinoa': 'others', 'seeds': 'others', 'sesame': 'others',
        'chocolate': 'others', 'candy': 'others', 'ice cream': 'others',
        'dessert': 'others', 'spice': 'others', 'seasoning': 'others',
        'salt': 'others', 'sugar': 'others'
    }

def map_to_7class(class_title, class_mapping):
    """Map original class to 7-class system"""
    class_title_lower = class_title.lower()
    
    # Try exact match first
    if class_title_lower in class_mapping:
        return class_mapping[class_title_lower]
    
    # Try substring matches
    for original_class, mapped_class in class_mapping.items():
        if original_class in class_title_lower or class_title_lower in original_class:
            return mapped_class
    
    # Default to others if no match found
    return 'others'

def process_rte8_data(data_root, output_dir, split, image_id_start, annotation_id_start, class_mapping):
    """Process RTE8 train/test data with bitmap masks"""
    
    split_dir = data_root / split
    if not split_dir.exists():
        return [], image_id_start, annotation_id_start
    
    img_dir = split_dir / 'img'
    ann_dir = split_dir / 'ann'
    
    if not img_dir.exists() or not ann_dir.exists():
        return [], image_id_start, annotation_id_start
    
    out_img_dir = output_dir / split / 'images'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    coco_data = []
    image_id = image_id_start
    annotation_id = annotation_id_start
    processed_count = 0
    
    print(f"Processing RTE8 {split} data...")
    
    for img_file in sorted(img_dir.glob('*.jpg')):
        ann_file = ann_dir / f"{img_file.name}.json"
        
        if not ann_file.exists():
            continue
        
        try:
            # Load image
            image = Image.open(img_file).convert('RGB')
            width, height = image.size
            
            # Copy image to output
            output_img_path = out_img_dir / f"{image_id:06d}.jpg"
            image.save(output_img_path)
            
            # Add image info
            image_data = {
                'id': image_id,
                'file_name': f"{image_id:06d}.jpg",
                'width': width,
                'height': height,
                'source': f'rte8_{split}',
                'original_name': img_file.name
            }
            
            # Load annotations
            with open(ann_file, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
            
            annotations = []
            
            # Process objects
            for obj in ann_data.get('objects', []):
                class_title = obj.get('classTitle', '')
                mapped_class = map_to_7class(class_title, class_mapping)
                category_id = ['carb', 'meat', 'vegetable', 'fruits', 'egg', 'gravy', 'others'].index(mapped_class) + 1
                
                # Handle bitmap geometry
                if obj.get('geometryType') == 'bitmap' and 'bitmap' in obj:
                    bitmap = obj['bitmap']
                    bitmap_data = bitmap['data']
                    origin = bitmap['origin']
                    
                    # Decode bitmap mask
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
                    area = int(np.sum(mask))
                    
                    # Create segmentation
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    segmentation = []
                    for contour in contours:
                        if len(contour) >= 3:
                            seg = contour.flatten().tolist()
                            if len(seg) >= 6:
                                segmentation.append(seg)
                    
                    if segmentation:
                        annotations.append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': [int(x_min), int(y_min), bbox_width, bbox_height],
                            'area': area,
                            'segmentation': segmentation,
                            'iscrowd': 0,
                            'original_class': class_title,
                            'mapped_class': mapped_class
                        })
                        annotation_id += 1
            
            if annotations:  # Only add images with valid annotations
                coco_data.append((image_data, annotations))
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} RTE8 images...")
            
            image_id += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"RTE8 {split} complete: {len(coco_data)} images")
    return coco_data, image_id, annotation_id

def merge_all_datasets_to_7class(data_root, output_dir):
    """Merge all datasets in initial folder and convert to 7-class COCO format"""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_mapping = get_7class_mapping()
    class_names = ['carb', 'meat', 'vegetable', 'fruits', 'egg', 'gravy', 'others']
    
    # Process both train and test splits
    for split in ['train', 'test']:
        # Create output structure
        out_split_dir = output_dir / split
        out_split_dir.mkdir(exist_ok=True)
        
        out_ann_dir = out_split_dir / 'annotations'
        out_ann_dir.mkdir(exist_ok=True)
        
        # Initialize COCO structure
        final_coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': i+1, 'name': name, 'supercategory': 'food'} 
                for i, name in enumerate(class_names)
            ]
        }
        
        image_id = 1
        annotation_id = 1
        
        # Process RTE8 data (main dataset with bitmap masks)
        rte8_data, image_id, annotation_id = process_rte8_data(
            data_root, output_dir, split, image_id, annotation_id, class_mapping
        )
        
        # Add RTE8 data to final dataset
        for image_data, annotations in rte8_data:
            final_coco_data['images'].append(image_data)
            final_coco_data['annotations'].extend(annotations)
        
        # TODO: Add Swiss validation data and Food dataset if they exist
        # These would require different parsing logic based on their formats
        
        # Save final COCO annotations
        with open(out_ann_dir / 'instances.json', 'w') as f:
            json.dump(final_coco_data, f, indent=2)
        
        print(f"\nCompleted {split} split:")
        print(f"  Total images: {len(final_coco_data['images'])}")
        print(f"  Total annotations: {len(final_coco_data['annotations'])}")
        
        # Print class distribution
        class_counts = defaultdict(int)
        for ann in final_coco_data['annotations']:
            cat_id = ann['category_id']
            class_name = class_names[cat_id - 1]
            class_counts[class_name] += 1
        
        print(f"  Class distribution:")
        total_objects = sum(class_counts.values())
        for class_name in class_names:
            count = class_counts[class_name]
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n7-class dataset conversion completed!")
    print(f"Dataset saved to: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    data_root = "../../data/initial"
    output_dir = "../data/7class_food_maskrcnn"
    
    if Path(data_root).exists():
        result_dir = merge_all_datasets_to_7class(data_root, output_dir)
        print(f"\nFinal dataset structure:")
        for split in ['train', 'test']:
            split_dir = result_dir / split
            if split_dir.exists():
                img_count = len(list((split_dir / 'images').glob('*.jpg')))
                print(f"  {split}: {img_count} images")
    else:
        print(f"Dataset not found at {data_root}")
        print("Please check the path and ensure the initial data folder exists.")