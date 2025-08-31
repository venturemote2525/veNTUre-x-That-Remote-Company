#!/usr/bin/env python3
"""
Log Management and Model Identification Script
Simplifies log file management and helps identify latest models
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

class LogManager:
    """Manages training logs and model checkpoints"""
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent
        else:
            project_root = Path(project_root)
            
        self.project_root = project_root
        self.logs_dir = project_root / "logs"
        self.checkpoints_dir = project_root / "checkpoints"
        
    def get_latest_logs(self, model_type: str = None) -> List[Dict]:
        """Get latest log files for each model type"""
        if not self.logs_dir.exists():
            return []
            
        logs = []
        
        # Scan all log directories
        for log_subdir in self.logs_dir.iterdir():
            if not log_subdir.is_dir():
                continue
                
            if model_type and model_type not in log_subdir.name:
                continue
                
            # Find latest log file in this directory
            log_files = list(log_subdir.glob("training_*.log"))
            if not log_files:
                continue
                
            # Sort by modification time
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            
            # Extract timestamp from filename
            timestamp_match = re.search(r'training_(\d{8}_\d{6})\.log', latest_log.name)
            timestamp_str = timestamp_match.group(1) if timestamp_match else "unknown"
            
            # Get file size and modification time
            stat = latest_log.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            file_size = stat.st_size
            
            # Parse log info
            log_info = self.parse_log_info(latest_log)
            
            logs.append({
                'model_type': log_subdir.name,
                'log_file': str(latest_log),
                'timestamp': timestamp_str,
                'mod_time': mod_time,
                'file_size': file_size,
                'status': log_info['status'],
                'current_epoch': log_info['current_epoch'],
                'total_epochs': log_info['total_epochs'],
                'latest_loss': log_info['latest_loss'],
                'is_running': log_info['is_running']
            })
        
        # Sort by modification time (latest first)
        logs.sort(key=lambda x: x['mod_time'], reverse=True)
        return logs
    
    def parse_log_info(self, log_file: Path) -> Dict:
        """Parse key information from log file"""
        info = {
            'status': 'unknown',
            'current_epoch': 0,
            'total_epochs': 0,
            'latest_loss': 0.0,
            'is_running': False
        }
        
        if not log_file.exists():
            return info
        
        try:
            # Read last few lines for current status
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if not lines:
                return info
            
            # Look for key patterns in recent lines
            recent_lines = lines[-100:]  # Last 100 lines
            
            for line in recent_lines:
                # Check if training is complete
                if "Two-stage training completed" in line:
                    info['status'] = 'completed'
                    info['is_running'] = False
                    
                # Extract current epoch
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                if epoch_match:
                    info['current_epoch'] = int(epoch_match.group(1))
                    info['total_epochs'] = int(epoch_match.group(2))
                    info['status'] = 'training'
                    
                # Extract latest loss
                loss_match = re.search(r'Average Training Loss: ([\d.]+)', line)
                if loss_match:
                    info['latest_loss'] = float(loss_match.group(1))
            
            # Check if file was modified recently (within 5 minutes)
            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            time_diff = (datetime.now() - mod_time).total_seconds()
            if time_diff < 300:  # 5 minutes
                info['is_running'] = True
            
        except Exception as e:
            print(f"Warning: Could not parse log file {log_file}: {e}")
        
        return info
    
    def get_latest_checkpoints(self) -> List[Dict]:
        """Get latest model checkpoints"""
        if not self.checkpoints_dir.exists():
            return []
        
        checkpoints = []
        
        for checkpoint_dir in self.checkpoints_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            # Look for checkpoint files
            checkpoint_files = []
            for ext in ['*.pth', '*.pt', '*.ckpt']:
                checkpoint_files.extend(list(checkpoint_dir.glob(ext)))
            
            if not checkpoint_files:
                continue
            
            # Find latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            # Get checkpoint info
            stat = latest_checkpoint.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            file_size = stat.st_size
            
            checkpoints.append({
                'model_type': checkpoint_dir.name,
                'checkpoint_file': str(latest_checkpoint),
                'mod_time': mod_time,
                'file_size': file_size,
                'size_mb': round(file_size / 1024 / 1024, 1)
            })
        
        # Sort by modification time (latest first)
        checkpoints.sort(key=lambda x: x['mod_time'], reverse=True)
        return checkpoints
    
    def create_simplified_log_links(self):
        """Create simplified symbolic links to latest logs"""
        simplified_dir = self.logs_dir / "latest"
        simplified_dir.mkdir(exist_ok=True)
        
        # Clear existing links
        for link in simplified_dir.glob("*"):
            if link.is_symlink() or link.is_file():
                link.unlink()
        
        # Create new links
        latest_logs = self.get_latest_logs()
        
        for log_info in latest_logs:
            model_type = log_info['model_type']
            log_file = Path(log_info['log_file'])
            
            # Create descriptive link name
            status = log_info['status']
            epoch = log_info['current_epoch']
            
            if status == 'training' and log_info['is_running']:
                link_name = f"{model_type}_RUNNING_epoch{epoch}.log"
            elif status == 'completed':
                link_name = f"{model_type}_COMPLETED.log"
            else:
                link_name = f"{model_type}_STOPPED_epoch{epoch}.log"
            
            link_path = simplified_dir / link_name
            
            try:
                # Create relative symlink
                rel_path = os.path.relpath(log_file, simplified_dir)
                os.symlink(rel_path, link_path)
            except (OSError, NotImplementedError):
                # Fallback: copy file instead of symlink
                import shutil
                shutil.copy2(log_file, link_path)
        
        print(f"Created simplified log links in: {simplified_dir}")
        return simplified_dir
    
    def print_summary(self):
        """Print summary of latest models and logs"""
        print("=" * 70)
        print("FOOD SEGMENTATION MODEL STATUS")
        print("=" * 70)
        
        print("\n📊 LATEST TRAINING LOGS:")
        print("-" * 40)
        
        latest_logs = self.get_latest_logs()
        
        if not latest_logs:
            print("No training logs found.")
        else:
            for log_info in latest_logs:
                model_type = log_info['model_type']
                status = log_info['status'].upper()
                epoch = log_info['current_epoch']
                total = log_info['total_epochs']
                loss = log_info['latest_loss']
                is_running = "🟢 RUNNING" if log_info['is_running'] else "⭕ STOPPED"
                
                print(f"\n{model_type}:")
                print(f"  Status: {status} {is_running}")
                if total > 0:
                    progress = (epoch / total) * 100
                    print(f"  Progress: {epoch}/{total} epochs ({progress:.1f}%)")
                if loss > 0:
                    print(f"  Latest Loss: {loss:.4f}")
                print(f"  Log: {log_info['log_file']}")
        
        print("\n💾 LATEST MODEL CHECKPOINTS:")
        print("-" * 40)
        
        latest_checkpoints = self.get_latest_checkpoints()
        
        if not latest_checkpoints:
            print("No model checkpoints found.")
        else:
            for checkpoint_info in latest_checkpoints:
                model_type = checkpoint_info['model_type']
                size_mb = checkpoint_info['size_mb']
                mod_time = checkpoint_info['mod_time'].strftime("%Y-%m-%d %H:%M")
                
                print(f"\n{model_type}:")
                print(f"  File: {checkpoint_info['checkpoint_file']}")
                print(f"  Size: {size_mb} MB")
                print(f"  Modified: {mod_time}")
        
        # Current recommendation
        print("\n🎯 CURRENT RECOMMENDATION:")
        print("-" * 40)
        
        running_models = [log for log in latest_logs if log['is_running']]
        if running_models:
            latest_running = running_models[0]
            print(f"✅ Currently training: {latest_running['model_type']}")
            print(f"   Progress: {latest_running['current_epoch']}/{latest_running['total_epochs']} epochs")
            print(f"   Log file: {latest_running['log_file']}")
        else:
            if latest_logs:
                latest_log = latest_logs[0]
                print(f"⚠️ No active training detected")
                print(f"   Latest model: {latest_log['model_type']}")
                print(f"   Status: {latest_log['status']} at epoch {latest_log['current_epoch']}")
            else:
                print("❌ No training logs found")

def main():
    parser = argparse.ArgumentParser(description='Manage training logs and model checkpoints')
    parser.add_argument('--model-type', type=str, help='Filter by model type (e.g., swiss_7class)')
    parser.add_argument('--create-links', action='store_true', help='Create simplified log links')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    manager = LogManager(args.project_root)
    
    if args.create_links:
        manager.create_simplified_log_links()
    
    manager.print_summary()

if __name__ == "__main__":
    main()