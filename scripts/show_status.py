#!/usr/bin/env python3
"""
Simple status display for training progress and latest models
"""

import os
import re
from pathlib import Path
from datetime import datetime

def find_latest_logs():
    """Find latest training logs"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    
    latest_logs = []
    
    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        log_files = list(subdir.glob("training_*.log"))
        if not log_files:
            continue
            
        # Get latest log file
        latest = max(log_files, key=lambda f: f.stat().st_mtime)
        
        # Check if currently running (modified within 5 minutes)
        mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
        is_running = (datetime.now() - mod_time).total_seconds() < 300
        
        # Parse current epoch
        current_epoch = 0
        total_epochs = 70
        latest_loss = 0.0
        
        try:
            with open(latest, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # Look for latest epoch info
            for line in reversed(lines[-50:]):  # Check last 50 lines
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))
                    break
                    
                loss_match = re.search(r'Average Training Loss: ([\d.]+)', line)
                if loss_match:
                    latest_loss = float(loss_match.group(1))
                    
        except Exception:
            pass
            
        latest_logs.append({
            'model': subdir.name,
            'file': str(latest),
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'latest_loss': latest_loss,
            'is_running': is_running,
            'mod_time': mod_time
        })
    
    # Sort by modification time (latest first)
    latest_logs.sort(key=lambda x: x['mod_time'], reverse=True)
    return latest_logs

def find_latest_checkpoints():
    """Find latest model checkpoints"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return []
    
    checkpoints = []
    
    for model_dir in checkpoints_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Look for checkpoint files
        checkpoint_files = []
        for pattern in ['*.pth', '*.pt', '*.ckpt']:
            checkpoint_files.extend(list(model_dir.glob(pattern)))
            
        if not checkpoint_files:
            continue
            
        # Get latest checkpoint
        latest = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
        size_mb = latest.stat().st_size / (1024 * 1024)
        
        checkpoints.append({
            'model': model_dir.name,
            'file': str(latest),
            'size_mb': round(size_mb, 1),
            'mod_time': mod_time
        })
    
    checkpoints.sort(key=lambda x: x['mod_time'], reverse=True)
    return checkpoints

def main():
    print("=" * 60)
    print("FOOD SEGMENTATION TRAINING STATUS")
    print("=" * 60)
    
    # Show training logs
    print("\nLATEST TRAINING LOGS:")
    print("-" * 40)
    
    logs = find_latest_logs()
    if not logs:
        print("No training logs found.")
    else:
        for log in logs:
            status = "RUNNING" if log['is_running'] else "STOPPED"
            progress = f"{log['current_epoch']}/{log['total_epochs']}" if log['total_epochs'] > 0 else "0/0"
            
            print(f"\n{log['model']}:")
            print(f"  Status: {status}")
            print(f"  Progress: {progress} epochs")
            if log['latest_loss'] > 0:
                print(f"  Latest Loss: {log['latest_loss']:.4f}")
            print(f"  Log File: {log['file']}")
    
    # Show checkpoints
    print("\nLATEST MODEL CHECKPOINTS:")
    print("-" * 40)
    
    checkpoints = find_latest_checkpoints()
    if not checkpoints:
        print("No model checkpoints found.")
    else:
        for checkpoint in checkpoints:
            print(f"\n{checkpoint['model']}:")
            print(f"  File: {checkpoint['file']}")
            print(f"  Size: {checkpoint['size_mb']} MB")
            print(f"  Modified: {checkpoint['mod_time'].strftime('%Y-%m-%d %H:%M')}")
    
    # Current recommendation
    print("\nRECOMMENDATION:")
    print("-" * 40)
    
    running_logs = [log for log in logs if log['is_running']]
    if running_logs:
        current = running_logs[0]
        print(f"Currently training: {current['model']}")
        print(f"Progress: {current['current_epoch']}/{current['total_epochs']} epochs")
        print(f"Log file: {current['file']}")
    else:
        if logs:
            latest = logs[0]
            print(f"No active training detected")
            print(f"Latest model: {latest['model']}")
            print(f"Stopped at epoch: {latest['current_epoch']}")
        else:
            print("No training found")

if __name__ == "__main__":
    main()