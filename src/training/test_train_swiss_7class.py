#!/usr/bin/env python3
"""
Test Train Mask R-CNN on Swiss Food 7-Class Dataset - 5 epochs with error logging
"""

import sys
import os
from pathlib import Path
import traceback

# Add pytorch_mask_rcnn to path
sys.path.append(str(Path(__file__).parent / "pytorch_mask_rcnn"))
sys.path.append(str(Path(__file__).parent / "pytorch_mask_rcnn" / "utils"))

from utils.config import FoodMaskRCNNConfig
from train_maskrcnn_food import FoodMaskRCNNTrainer
import torch
import time

def main():
    try:
        print("=" * 60)
        print("TEST TRAINING - Swiss Food 7-Class Dataset")
        print("5 epochs total with comprehensive error logging")
        print("=" * 60)
        
        # Swiss 7-class configuration
        swiss_7_classes = ['fruit', 'vegetable', 'carbohydrate', 'protein', 'dairy', 'fat', 'other']
        
        # Create config with TEST parameters
        config = FoodMaskRCNNConfig(
            name='swiss_7class_test',
            data_root='data/swiss_coco_7class',
            epochs=5,  # Small test run
            batch_size=1,
            learning_rate=0.005,
            class_names=swiss_7_classes,
            num_workers=0,
            heads_epochs=2  # Stage 1: 2 epochs, Stage 2: 3 epochs
        )
        
        print(f"\n7 Food Classes: {swiss_7_classes}")
        print(f"Total classes (including background): {config.num_classes}")
        print(f"Total epochs: {config.epochs}")
        print(f"Stage 1 (heads): {getattr(config, 'heads_epochs', 2)} epochs")
        print(f"Stage 2 (all): {config.epochs - getattr(config, 'heads_epochs', 2)} epochs")
        
        # Create trainer with enhanced error handling
        trainer = TestFoodMaskRCNNTrainer(config)
        
        # Build model
        print("\n" + "="*30)
        print("BUILDING MODEL")
        print("="*30)
        trainer.build_model()
        
        # Setup data loaders
        print("\n" + "="*30)
        print("SETTING UP DATA LOADERS")
        print("="*30)
        trainer.setup_data_loaders()
        
        # Setup optimizer
        print("\n" + "="*30)
        print("SETTING UP OPTIMIZER")
        print("="*30)
        trainer.setup_optimizer()
        
        # Start training with comprehensive error logging
        print("\n" + "="*30)
        print("STARTING TRAINING")
        print("="*30)
        trainer.train()
        
        print("\n" + "="*60)
        print("TEST TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n" + "="*60)
        print("CRITICAL ERROR DURING TEST TRAINING!")
        print("="*60)
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Clear CUDA cache if OOM
        if "out of memory" in str(e).lower():
            print("\nClearing CUDA cache due to OOM error...")
            torch.cuda.empty_cache()
        
        sys.exit(1)

class TestFoodMaskRCNNTrainer(FoodMaskRCNNTrainer):
    """Enhanced trainer with comprehensive error logging"""
    
    def train(self):
        """Two-stage training with comprehensive error logging and crash protection"""
        print(f"\n{'='*50}")
        print("Starting Enhanced Two-Stage Training")
        print(f"{'='*50}")
        
        best_val_map = 0.0
        start_time = time.time()
        
        try:
            # Stage 1: Train heads only
            print(f"\n{'='*20} STAGE 1: HEADS ONLY {'='*20}")
            self.logger.info("Starting Stage 1: Heads-only training")
            
            self.freeze_backbone()
            
            # Adjust learning rate for heads training
            heads_lr = self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = heads_lr
            
            heads_epochs = getattr(self.config, 'heads_epochs', max(1, self.config.epochs // 3))
            print(f"Training heads for {heads_epochs} epochs with LR: {heads_lr}")
            self.logger.info(f"Stage 1: {heads_epochs} epochs, LR: {heads_lr}")
            
            # Stage 1 training loop
            for epoch in range(heads_epochs):
                try:
                    print(f"\nStage 1 - Epoch {epoch+1}/{heads_epochs}")
                    print("-" * 30)
                    self.logger.info(f"Stage 1 - Epoch {epoch+1}/{heads_epochs} starting")
                    
                    # Train
                    train_loss = self.train_epoch(epoch)
                    
                    # Validate
                    val_loss, val_map = self.validate_epoch(epoch)
                    
                    # Step scheduler
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Save checkpoint
                    is_best = val_map > best_val_map
                    if is_best:
                        best_val_map = val_map
                    
                    self.save_checkpoint(epoch, val_map, is_best)
                    
                    # Print epoch summary
                    current_lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    print(f"Stage 1 Summary:")
                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}")
                    print(f"  Val mAP: {val_map:.4f}")
                    print(f"  Best mAP: {best_val_map:.4f}")
                    print(f"  Learning Rate: {current_lr:.6f}")
                    print(f"  Elapsed Time: {elapsed/3600:.2f}h")
                    
                    self.logger.info(f"Stage 1 Epoch {epoch+1} completed - Loss: {train_loss:.4f}, mAP: {val_map:.4f}")
                    
                except Exception as e:
                    error_msg = f"Error in Stage 1, Epoch {epoch+1}: {e}"
                    print(f"ERROR: {error_msg}")
                    self.logger.error(error_msg)
                    self.logger.error(traceback.format_exc())
                    raise
            
            print(f"\n{'='*20} STAGE 1 COMPLETED {'='*20}")
            self.logger.info("Stage 1 completed successfully")
            
            # Clear cache before stage transition
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache before Stage 2")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR in Stage 1: {e}"
            print(f"ERROR: {error_msg}")
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise
        
        try:
            # Stage 2: Fine-tune all layers
            print(f"\n{'='*20} STAGE 2: ALL LAYERS {'='*20}")
            self.logger.info("Starting Stage 2: Full fine-tuning")
            
            # Unfreeze with error handling
            try:
                self.unfreeze_all_layers()
                print("Successfully unfroze all layers")
                self.logger.info("All layers unfrozen successfully")
            except Exception as e:
                error_msg = f"Error unfreezing layers: {e}"
                print(f"ERROR: {error_msg}")
                self.logger.error(error_msg)
                raise
            
            # Lower learning rate for full fine-tuning
            finetune_lr = self.config.learning_rate / 10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = finetune_lr
            
            # Reset scheduler for stage 2
            remaining_epochs = self.config.epochs - heads_epochs
            if remaining_epochs <= 0:
                print("WARNING: No epochs remaining for Stage 2!")
                self.logger.warning("No epochs remaining for Stage 2")
                return
                
            try:
                if self.config.scheduler == 'step':
                    self.scheduler = torch.optim.lr_scheduler.StepLR(
                        self.optimizer,
                        step_size=self.config.lr_step_size,
                        gamma=self.config.lr_gamma
                    )
                elif self.config.scheduler == 'cosine':
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=remaining_epochs
                    )
                print("Scheduler reset successfully")
                self.logger.info("Scheduler reset for Stage 2")
            except Exception as e:
                error_msg = f"Error resetting scheduler: {e}"
                print(f"ERROR: {error_msg}")
                self.logger.error(error_msg)
                raise
            
            print(f"Fine-tuning all layers for {remaining_epochs} epochs with LR: {finetune_lr}")
            self.logger.info(f"Stage 2: {remaining_epochs} epochs, LR: {finetune_lr}")
            
            # Stage 2 training loop
            for epoch in range(heads_epochs, self.config.epochs):
                try:
                    print(f"\nStage 2 - Epoch {epoch+1}/{self.config.epochs}")
                    print("-" * 30)
                    self.logger.info(f"Stage 2 - Epoch {epoch+1}/{self.config.epochs} starting")
                    
                    # Train
                    train_loss = self.train_epoch(epoch)
                    
                    # Validate
                    val_loss, val_map = self.validate_epoch(epoch)
                    
                    # Step scheduler
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Save checkpoint
                    is_best = val_map > best_val_map
                    if is_best:
                        best_val_map = val_map
                    
                    self.save_checkpoint(epoch, val_map, is_best)
                    
                    # Save training history
                    self.save_training_history()
                    
                    # Print epoch summary
                    current_lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    print(f"Stage 2 Summary:")
                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}")
                    print(f"  Val mAP: {val_map:.4f}")
                    print(f"  Best mAP: {best_val_map:.4f}")
                    print(f"  Learning Rate: {current_lr:.6f}")
                    print(f"  Elapsed Time: {elapsed/3600:.2f}h")
                    
                    self.logger.info(f"Stage 2 Epoch {epoch+1} completed - Loss: {train_loss:.4f}, mAP: {val_map:.4f}")
                    
                except Exception as e:
                    error_msg = f"Error in Stage 2, Epoch {epoch+1}: {e}"
                    print(f"ERROR: {error_msg}")
                    self.logger.error(error_msg)
                    self.logger.error(traceback.format_exc())
                    raise
            
            print(f"\n{'='*20} STAGE 2 COMPLETED {'='*20}")
            self.logger.info("Stage 2 completed successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR in Stage 2: {e}"
            print(f"ERROR: {error_msg}")
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Two-stage training completed in {total_time/3600:.2f} hours")
        print(f"Final validation mAP: {best_val_map:.4f}")
        print(f"Stage 1 (heads): {heads_epochs} epochs")
        print(f"Stage 2 (all): {remaining_epochs} epochs")
        print(f"{'='*50}")
        
        self.logger.info(f"Training completed successfully - Total time: {total_time/3600:.2f}h, Final mAP: {best_val_map:.4f}")

if __name__ == "__main__":
    main()