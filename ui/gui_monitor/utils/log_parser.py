#!/usr/bin/env python3
"""
Log Parser Utility for Training Monitor
Parses training logs and extracts training progress data
"""
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class LogParser:
    """Parse training logs to extract progress information."""
    
    def __init__(self):
        self.epoch_pattern = re.compile(r'Epoch (\d+)')
        self.loss_pattern = re.compile(r'Loss: ([\d\.]+)')
        self.train_loss_pattern = re.compile(r'Train Loss: ([\d\.]+)')
        self.val_loss_pattern = re.compile(r'Val Loss: ([\d\.]+)')
        self.batch_pattern = re.compile(r'Batch (\d+)/(\d+)')
        self.timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    
    def parse_log(self, log_file_path: str) -> Dict:
        """
        Parse training log file and extract progress data.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Dictionary containing parsed training data
        """
        if not os.path.exists(log_file_path):
            return {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'batch_progress': [],
                'timestamps': [],
                'status': 'Log file not found'
            }
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            return {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'batch_progress': [],
                'timestamps': [],
                'status': f'Error reading file: {str(e)}'
            }
        
        # Parse data
        epochs = []
        train_losses = []
        val_losses = []
        batch_progress = []
        timestamps = []
        current_epoch = 0
        current_batch = 0
        total_batches = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract timestamp
            timestamp_match = self.timestamp_pattern.search(line)
            if timestamp_match:
                try:
                    timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    timestamps.append(timestamp)
                except ValueError:
                    pass
            
            # Extract epoch information
            epoch_match = self.epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if current_epoch not in epochs:
                    epochs.append(current_epoch)
            
            # Extract batch information
            batch_match = self.batch_pattern.search(line)
            if batch_match:
                current_batch = int(batch_match.group(1))
                total_batches = int(batch_match.group(2))
                if total_batches > 0:
                    progress = (current_batch / total_batches) * 100
                    batch_progress.append(progress)
            
            # Extract loss information
            train_loss_match = self.train_loss_pattern.search(line)
            if train_loss_match:
                train_loss = float(train_loss_match.group(1))
                train_losses.append(train_loss)
            
            val_loss_match = self.val_loss_pattern.search(line)
            if val_loss_match:
                val_loss = float(val_loss_match.group(1))
                val_losses.append(val_loss)
            
            # Extract generic loss (for batch-level losses)
            if 'Loss:' in line and not any(x in line for x in ['Train Loss:', 'Val Loss:']):
                loss_match = self.loss_pattern.search(line)
                if loss_match:
                    loss_value = float(loss_match.group(1))
                    # This could be batch-level training loss
                    if len(train_losses) == 0 or abs(loss_value - train_losses[-1]) > 0.001:
                        train_losses.append(loss_value)
        
        # Determine training status
        status = "Unknown"
        if epochs:
            if len(lines) > 0:
                last_line = lines[-1].lower()
                if any(keyword in last_line for keyword in ['completed', 'finished', 'done']):
                    status = "Completed"
                elif any(keyword in last_line for keyword in ['error', 'failed', 'exception']):
                    status = "Failed"
                elif any(keyword in last_line for keyword in ['epoch', 'batch', 'loss']):
                    status = "Training"
                else:
                    status = "Active"
            else:
                status = "Active"
        else:
            status = "Initializing"
        
        # Align data lengths
        max_epochs = len(epochs)
        if len(train_losses) > max_epochs:
            # Take average losses per epoch if we have more losses than epochs
            epoch_losses = []
            losses_per_epoch = len(train_losses) // max_epochs if max_epochs > 0 else len(train_losses)
            for i in range(max_epochs):
                start_idx = i * losses_per_epoch
                end_idx = min(start_idx + losses_per_epoch, len(train_losses))
                if start_idx < len(train_losses):
                    epoch_loss = sum(train_losses[start_idx:end_idx]) / max(1, end_idx - start_idx)
                    epoch_losses.append(epoch_loss)
            train_losses = epoch_losses
        
        # Ensure we have enough timestamps
        if len(timestamps) < len(epochs):
            # Generate missing timestamps based on file modification time
            try:
                file_mtime = os.path.getmtime(log_file_path)
                base_time = datetime.fromtimestamp(file_mtime)
                for i in range(len(timestamps), len(epochs)):
                    # Estimate timestamp (assume 2 minutes per epoch)
                    estimated_time = base_time - timedelta(minutes=(len(epochs) - i - 1) * 2)
                    timestamps.append(estimated_time)
            except:
                # Fallback: use current time with offsets
                from datetime import timedelta
                base_time = datetime.now()
                for i in range(len(timestamps), len(epochs)):
                    estimated_time = base_time - timedelta(minutes=(len(epochs) - i - 1) * 2)
                    timestamps.append(estimated_time)
        
        return {
            'epochs': epochs,
            'train_loss': train_losses[:len(epochs)],
            'val_loss': val_losses[:len(epochs)],
            'batch_progress': batch_progress,
            'timestamps': timestamps[:len(epochs)],
            'status': status,
            'current_epoch': current_epoch,
            'total_batches': total_batches,
            'current_batch': current_batch
        }
    
    def get_latest_stats(self, log_file_path: str) -> Dict:
        """
        Get latest training statistics from log file.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Dictionary with latest stats
        """
        data = self.parse_log(log_file_path)
        
        latest_stats = {
            'status': data.get('status', 'Unknown'),
            'current_epoch': data.get('current_epoch', 0),
            'total_epochs': len(data.get('epochs', [])),
            'latest_train_loss': data['train_loss'][-1] if data.get('train_loss') else None,
            'latest_val_loss': data['val_loss'][-1] if data.get('val_loss') else None,
            'progress_percent': 0
        }
        
        if data.get('total_batches', 0) > 0 and data.get('current_batch', 0) > 0:
            latest_stats['progress_percent'] = (data['current_batch'] / data['total_batches']) * 100
        
        return latest_stats
    
    def extract_training_config(self, log_file_path: str) -> Dict:
        """
        Extract training configuration from log file.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Dictionary with training configuration
        """
        if not os.path.exists(log_file_path):
            return {}
        
        config = {}
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line in lines[:50]:  # Check first 50 lines for config
                line = line.strip()
                
                # Extract model info
                if 'Model parameters:' in line:
                    match = re.search(r'Model parameters: ([\d,]+)', line)
                    if match:
                        config['total_parameters'] = match.group(1)
                
                if 'Trainable parameters:' in line:
                    match = re.search(r'Trainable parameters: ([\d,]+)', line)
                    if match:
                        config['trainable_parameters'] = match.group(1)
                
                # Extract batch size
                if 'Batch Size:' in line:
                    match = re.search(r'Batch Size: (\d+)', line)
                    if match:
                        config['batch_size'] = int(match.group(1))
                
                # Extract workers
                if 'Workers:' in line:
                    match = re.search(r'Workers: (\d+)', line)
                    if match:
                        config['num_workers'] = int(match.group(1))
                
                # Extract backbone
                if 'Backbone:' in line:
                    match = re.search(r'Backbone: (\w+)', line)
                    if match:
                        config['backbone'] = match.group(1)
                
                # Extract GPU info
                if 'GPU:' in line:
                    match = re.search(r'GPU: (.+)', line)
                    if match:
                        config['gpu'] = match.group(1).strip()
                
                if 'GPU Memory:' in line:
                    match = re.search(r'GPU Memory: ([\d\.]+) GB', line)
                    if match:
                        config['gpu_memory'] = f"{match.group(1)} GB"
        
        except Exception as e:
            config['error'] = f"Error parsing config: {str(e)}"
        
        return config