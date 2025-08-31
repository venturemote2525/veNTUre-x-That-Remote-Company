#!/usr/bin/env python3
"""
Food Segmentation Dataset for PyTorch Mask R-CNN
Supports COCO format annotations
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from pycocotools import mask as coco_mask
from pathlib import Path
import torchvision.transforms as T

class FoodSegmentationDataset(Dataset):
    """Food segmentation dataset in COCO format"""
    
    def __init__(self, root_dir, split='train', transforms=None, class_names=None):
        """
        Args:
            root_dir: Root directory containing images and annotations
            split: Dataset split ('train', 'val', 'test')
            transforms: Image transforms
            class_names: List of class names
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        
        # Set up paths - handle both COCO format and RTE8 format
        original_split = self.split
        if self.split == 'val':
            # Check if val exists, otherwise use test split as validation for RTE8 dataset
            if not (self.root_dir / 'val').exists() and (self.root_dir / 'test').exists():
                self.split = 'test'
            
        if (self.root_dir / self.split / 'images').exists():
            # COCO format
            self.img_dir = self.root_dir / self.split / 'images'
            self.ann_file = self.root_dir / self.split / 'annotations' / 'instances.json'
        else:
            # RTE8 format
            self.img_dir = self.root_dir / self.split / 'img'
            self.ann_file = self.root_dir / self.split / 'annotations' / 'instances.json'
        
        # Check if annotation file exists
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
            
        print(f"Using {original_split} split (mapped to {self.split}): {self.ann_file}")
        
        # Load COCO annotations
        print(f"Loading {split} annotations from {self.ann_file}")
        with open(self.ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Image IDs list
        self.image_ids = list(self.images.keys())
        
        # Class names
        if class_names is not None:
            self.class_names = ['background'] + class_names
        else:
            self.class_names = ['background'] + [cat['name'] for cat in self.categories.values()]
        
        self.num_classes = len(self.class_names)
        
        print(f"Loaded {len(self.image_ids)} images")
        print(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get dataset item"""
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Prepare target
        target = {}
        target['image_id'] = torch.tensor([img_id])
        
        if len(anns) == 0:
            # No annotations - return empty tensors
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.uint8)
        else:
            # Process annotations
            boxes = []
            labels = []
            masks = []
            areas = []
            iscrowd = []
            
            for ann in anns:
                # Bounding box
                bbox = ann['bbox']  # [x, y, width, height]
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                boxes.append([x1, y1, x2, y2])
                
                # Label (category_id -> class index)
                cat_id = ann['category_id']
                # Map category_id to class index (1-based, 0 is background)
                # For rte8_coco_quick dataset: map [1,2,3,7] -> [1,2,3,4] 
                if cat_id == 7:  # 'others' class
                    label = 4
                else:
                    label = cat_id  # [1,2,3] stay as [1,2,3]
                labels.append(label)
                
                # Mask with memory optimization
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format with memory optimization
                        try:
                            rle = coco_mask.frPyObjects(ann['segmentation'], 
                                                       img_info['height'], 
                                                       img_info['width'])
                            mask = coco_mask.decode(rle)
                            if len(mask.shape) == 3:
                                # Use more memory-efficient merging
                                mask = np.any(mask, axis=2).astype(np.uint8)
                            else:
                                mask = (mask > 0).astype(np.uint8)
                        except (MemoryError, np.core._exceptions._ArrayMemoryError):
                            # Fallback: create mask from bbox if memory error
                            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                            mask[int(y1):int(y2), int(x1):int(x2)] = 1
                    else:
                        # RLE format
                        if isinstance(ann['segmentation']['counts'], list):
                            rle = coco_mask.frPyObjects([ann['segmentation']], 
                                                       img_info['height'], 
                                                       img_info['width'])
                        else:
                            rle = [ann['segmentation']]
                        mask = coco_mask.decode(rle)[..., 0]
                else:
                    # Create mask from bbox if no segmentation
                    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                    mask[int(y1):int(y2), int(x1):int(x2)] = 1
                
                masks.append(mask)
                
                # Area and iscrowd
                areas.append(ann.get('area', w * h))
                iscrowd.append(ann.get('iscrowd', 0))
            
            # Convert to tensors
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['masks'] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            target['area'] = torch.as_tensor(areas, dtype=torch.float32)
            target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.uint8)
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            # Default: convert PIL to tensor
            image = T.functional.to_tensor(image)
        
        return image, target

class FoodDatasetTransforms:
    """Data transforms for food dataset"""
    
    def __init__(self, training=True):
        self.training = training
    
    def __call__(self, image, target):
        # Convert PIL to tensor
        image = T.functional.to_tensor(image)
        
        # Training augmentations
        if self.training:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                image = T.functional.hflip(image)
                if target['boxes'].numel() > 0:
                    # Flip boxes
                    _, _, width = image.shape
                    target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
                    # Flip masks
                    target['masks'] = torch.flip(target['masks'], [2])
        
        return image, target

# Utility functions
def create_food_datasets(data_root, class_names=None):
    """Create train and val datasets"""
    
    train_transforms = FoodDatasetTransforms(training=True)
    val_transforms = FoodDatasetTransforms(training=False)
    
    train_dataset = FoodSegmentationDataset(
        root_dir=data_root,
        split='train',
        transforms=train_transforms,
        class_names=class_names
    )
    
    val_dataset = FoodSegmentationDataset(
        root_dir=data_root,
        split='val',
        transforms=val_transforms,
        class_names=class_names
    )
    
    return train_dataset, val_dataset

def visualize_sample(dataset, idx, save_path=None):
    """Visualize a dataset sample"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon
    
    image, target = dataset[idx]
    
    # Convert tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Image with annotations
    axes[1].imshow(image)
    
    if target['boxes'].numel() > 0:
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        masks = target['masks'].numpy()
        
        for i in range(len(boxes)):
            # Bounding box
            x1, y1, x2, y2 = boxes[i]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect)
            
            # Label
            label_name = dataset.class_names[labels[i]] if labels[i] < len(dataset.class_names) else f"class_{labels[i]}"
            axes[1].text(x1, y1-5, label_name, color='red', fontsize=10, weight='bold')
            
            # Mask outline
            mask = masks[i]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze(1)
                if len(contour) > 2:
                    polygon = Polygon(contour, closed=True, fill=False, 
                                    edgecolor='yellow', linewidth=2, alpha=0.8)
                    axes[1].add_patch(polygon)
    
    axes[1].set_title('Annotations')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    # Test the dataset
    data_root = "../data/merged/rte8_7class_food_dataset"
    class_names = ["carb", "meat", "vegetable", "others"]
    
    if Path(data_root).exists():
        dataset = FoodSegmentationDataset(
            root_dir=data_root,
            split='train',
            class_names=class_names
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test first sample
            image, target = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Number of objects: {len(target['boxes'])}")
            print(f"Labels: {target['labels']}")
            
            # Visualize
            fig = visualize_sample(dataset, 0, "sample_visualization.png")
        else:
            print("Dataset is empty!")
    else:
        print(f"Data root {data_root} does not exist!")