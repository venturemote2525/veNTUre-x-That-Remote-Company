#!/usr/bin/env python3
"""
Quick converter for RTE8 dataset to COCO format for immediate training
"""

import json
import os
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import shutil

def create_simple_coco_from_rte8(rte8_root, output_dir):
    """Create simple COCO format from existing RTE8 data"""
    
    rte8_root = Path(rte8_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 7-class mapping (simplified)
    class_names = ["carb", "meat", "vegetable", "fruits", "egg", "gravy", "others"]
    
    for split in ['train', 'test']:
        split_dir = rte8_root / split
        if not split_dir.exists():
            continue
            
        img_dir = split_dir / 'img'
        ann_dir = split_dir / 'ann'
        
        if not img_dir.exists():
            continue
        
        # Create output structure
        out_split_dir = output_dir / split
        out_split_dir.mkdir(exist_ok=True)
        
        out_img_dir = out_split_dir / 'images'
        out_img_dir.mkdir(exist_ok=True)
        
        out_ann_dir = out_split_dir / 'annotations'
        out_ann_dir.mkdir(exist_ok=True)
        
        # Simple COCO structure
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
        
        # Process images (without complex mask decoding for now)
        for img_file in sorted(list(img_dir.glob('*.jpg'))[:100]):  # Limit to 100 for quick start
            try:
                # Load and copy image
                image = Image.open(img_file).convert('RGB')
                width, height = image.size
                
                output_img_path = out_img_dir / f"{image_id:06d}.jpg"
                image.save(output_img_path)
                
                # Add image info
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': f"{image_id:06d}.jpg",
                    'width': width,
                    'height': height
                })
                
                # Create dummy annotation for now (will be replaced with real data)
                ann_file = ann_dir / f"{img_file.name}.json"
                if ann_file.exists():
                    try:
                        with open(ann_file, 'r', encoding='utf-8') as f:
                            ann_data = json.load(f)
                        
                        for obj in ann_data.get('objects', [])[:3]:  # Max 3 objects per image
                            class_title = obj.get('classTitle', '').lower()
                            
                            # Simple class mapping
                            if any(x in class_title for x in ['rice', 'bread', 'noodle', 'pasta', 'potato']):
                                category_id = 1  # carb
                            elif any(x in class_title for x in ['chicken', 'meat', 'beef', 'pork', 'fish']):
                                category_id = 2  # meat
                            elif any(x in class_title for x in ['carrot', 'vegetable', 'onion', 'cucumber']):
                                category_id = 3  # vegetable
                            else:
                                category_id = 7  # others
                            
                            # Create simple bounding box (center of image for now)
                            x = width // 4
                            y = height // 4
                            w = width // 2
                            h = height // 2
                            
                            # Simple segmentation (rectangle)
                            segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                            
                            coco_data['annotations'].append({
                                'id': annotation_id,
                                'image_id': image_id,
                                'category_id': category_id,
                                'bbox': [x, y, w, h],
                                'area': w * h,
                                'segmentation': segmentation,
                                'iscrowd': 0
                            })
                            
                            annotation_id += 1
                    except:
                        pass
                
                image_id += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        # Save annotations
        with open(out_ann_dir / 'instances.json', 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Created {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    return output_dir

if __name__ == "__main__":
    rte8_root = "../../data/initial"
    output_dir = "../data/rte8_coco_quick"
    
    if Path(rte8_root).exists():
        result_dir = create_simple_coco_from_rte8(rte8_root, output_dir)
        print(f"Quick COCO dataset created at: {result_dir}")
    else:
        print(f"RTE8 data not found at {rte8_root}")