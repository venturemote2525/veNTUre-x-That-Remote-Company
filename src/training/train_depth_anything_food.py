#!/usr/bin/env python3
"""
Train Depth Anything on Swiss Food 7-Class Dataset for Food Volume Estimation
Integrates depth estimation with food segmentation for portion size analysis
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import logging

# Add depth_anything to path
depth_anything_path = Path(__file__).parent.parent / "models" / "depth_anything"
sys.path.append(str(depth_anything_path))

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class FoodDepthEstimator:
    """
    Depth estimation for food images using Depth Anything
    Adapted for food portion size estimation with 7-class segmentation
    """
    
    def __init__(self, encoder='vitb', device='cuda'):
        self.device = device
        self.encoder = encoder
        
        # Load pretrained Depth Anything model
        self.model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(device).eval()
        
        # Swiss 7-class food categories
        self.food_classes = ['fruit', 'vegetable', 'carbohydrate', 'protein', 'dairy', 'fat', 'other']
        
        # Transform for depth estimation
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for training/inference"""
        log_dir = Path(__file__).parent.parent.parent / "logs" / "depth_anything_food"
        log_dir.mkdir(exist_ok=True, parents=True)
        
        log_file = log_dir / f"depth_estimation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('FoodDepthEstimator')
        self.logger.info(f"Initialized Food Depth Estimator with {self.encoder} encoder")
        
    def estimate_depth(self, image_path, output_dir=None):
        """
        Estimate depth for a single food image
        
        Args:
            image_path: Path to input image
            output_dir: Optional directory to save depth visualization
            
        Returns:
            depth_map: Numpy array of depth values
        """
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        h, w = image.shape[:2]
        
        # Transform image for depth estimation
        image_tensor = self.transform({'image': image / 255.0})['image']
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(self.device)
        
        # Estimate depth
        with torch.no_grad():
            depth = self.model(image_tensor)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = depth.cpu().numpy()
        
        # Save visualization if requested
        if output_dir:
            self.save_depth_visualization(image, depth, image_path, output_dir)
            
        return depth
    
    def save_depth_visualization(self, original_image, depth_map, image_path, output_dir):
        """Save depth visualization with original image side-by-side"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Normalize depth for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # Create side-by-side visualization
        h, w = original_image.shape[:2]
        combined = np.zeros((h, w * 2 + 50, 3), dtype=np.uint8)
        combined[:, :w] = original_image
        combined[:, w+50:] = depth_colored
        
        # Add labels
        cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'Depth', (w + 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        output_file = output_dir / f"{Path(image_path).stem}_depth.jpg"
        cv2.imwrite(str(output_file), combined)
        
        self.logger.info(f"Saved depth visualization: {output_file}")
    
    def process_food_dataset(self, dataset_root, output_dir, split='val'):
        """
        Process Swiss 7-class food dataset for depth estimation
        
        Args:
            dataset_root: Path to swiss_coco_7class dataset
            output_dir: Directory to save results
            split: Dataset split ('train', 'val', or 'test')
        """
        dataset_path = Path(dataset_root) / split
        images_dir = dataset_path / 'images'
        annotations_file = dataset_path / 'annotations' / 'instances.json'
        
        if not images_dir.exists() or not annotations_file.exists():
            raise ValueError(f"Dataset not found at {dataset_path}")
            
        # Load annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
            
        # Create output directories
        output_dir = Path(output_dir)
        depth_dir = output_dir / 'depth_maps'
        vis_dir = output_dir / 'visualizations'
        results_dir = output_dir / 'results'
        
        for dir_path in [depth_dir, vis_dir, results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Process images
        results = []
        for image_info in tqdm(coco_data['images'], desc=f"Processing {split} images"):
            image_path = images_dir / image_info['file_name']
            
            try:
                # Estimate depth
                depth_map = self.estimate_depth(image_path, vis_dir)
                
                # Save raw depth data
                depth_file = depth_dir / f"{image_info['id']:06d}.npy"
                np.save(depth_file, depth_map)
                
                # Analyze depth statistics
                depth_stats = {
                    'image_id': image_info['id'],
                    'file_name': image_info['file_name'],
                    'depth_min': float(depth_map.min()),
                    'depth_max': float(depth_map.max()),
                    'depth_mean': float(depth_map.mean()),
                    'depth_std': float(depth_map.std())
                }
                
                results.append(depth_stats)
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                continue
        
        # Save results
        results_file = results_dir / f'{split}_depth_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Processed {len(results)} images from {split} split")
        self.logger.info(f"Results saved to {results_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Food Depth Estimation with Depth Anything')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Path to swiss_coco_7class dataset')
    parser.add_argument('--output-dir', type=str, default='./food_depth_results',
                       help='Output directory for results')
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl'],
                       help='Depth Anything encoder size')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--single-image', type=str,
                       help='Process single image instead of dataset')
    
    args = parser.parse_args()
    
    # Initialize estimator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = FoodDepthEstimator(encoder=args.encoder, device=device)
    
    print(f"\nFood Depth Estimation with Depth Anything")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {device}")
    print(f"Food Classes: {estimator.food_classes}")
    print("-" * 50)
    
    if args.single_image:
        # Process single image
        depth_map = estimator.estimate_depth(args.single_image, args.output_dir)
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: {depth_map.min():.3f} to {depth_map.max():.3f}")
        
    else:
        # Process dataset
        results = estimator.process_food_dataset(
            args.dataset_root, 
            args.output_dir, 
            args.split
        )
        
        print(f"\nDataset Processing Complete:")
        print(f"Processed {len(results)} images")
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()