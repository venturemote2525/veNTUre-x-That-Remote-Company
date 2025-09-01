#!/usr/bin/env python3
"""
Real-time training monitor for Mask R-CNN training
Displays logs from both console output and log files
"""

import os
import sys
import time
import threading
from pathlib import Path
import subprocess
from datetime import datetime

class TrainingMonitor:
    def __init__(self, log_dir="logs", config_name="food_7class"):
        self.log_dir = Path(log_dir) / config_name
        self.log_files = []
        self.running = True
        self.last_positions = {}
        
    def find_latest_log(self):
        """Find the most recent training log file"""
        if not self.log_dir.exists():
            return None
            
        log_files = list(self.log_dir.glob("training_*.log"))
        if not log_files:
            return None
            
        # Get the most recent log file
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        return latest_log
        
    def tail_log_file(self, log_file):
        """Tail a log file and display new content"""
        if not log_file.exists():
            return
            
        # Get file size for tracking new content
        if str(log_file) not in self.last_positions:
            self.last_positions[str(log_file)] = 0
            
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                # Seek to last position
                f.seek(self.last_positions[str(log_file)])
                
                # Read new lines
                new_lines = f.readlines()
                if new_lines:
                    print(f"\n[LOG UPDATE - {log_file.name}]")
                    for line in new_lines:
                        # Format log lines for better readability
                        line = line.strip()
                        if line:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            if "ERROR" in line or "error" in line:
                                print(f"[ERROR] [{timestamp}] {line}")
                            elif "WARNING" in line or "warning" in line:
                                print(f"[WARN] [{timestamp}] {line}")
                            elif "INFO" in line:
                                print(f"[INFO] [{timestamp}] {line}")
                            else:
                                print(f"[LOG] [{timestamp}] {line}")
                    print("=" * 60)
                
                # Update position
                self.last_positions[str(log_file)] = f.tell()
                
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
            
    def monitor_gpu(self):
        """Monitor GPU usage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,nounits,noheader'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                if len(gpu_info) >= 4:
                    util, mem_used, mem_total, temp = [x.strip() for x in gpu_info]
                    mem_percent = (int(mem_used) / int(mem_total)) * 100
                    return f"GPU: {util}% util, {mem_used}MB/{mem_total}MB ({mem_percent:.1f}%), {temp}°C"
        except:
            pass
        return "GPU: monitoring unavailable"
        
    def monitor_logs(self):
        """Main monitoring loop"""
        print("[TRAINING MONITOR] Starting Training Monitor")
        print("=" * 60)
        print(f"[MONITOR] Monitoring directory: {self.log_dir}")
        print(f"[MONITOR] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        last_gpu_update = 0
        
        while self.running:
            try:
                # Find and tail latest log
                latest_log = self.find_latest_log()
                if latest_log:
                    self.tail_log_file(latest_log)
                
                # Update GPU info every 30 seconds
                current_time = time.time()
                if current_time - last_gpu_update > 30:
                    gpu_info = self.monitor_gpu()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[GPU] [{timestamp}] {gpu_info}")
                    last_gpu_update = current_time
                
                # Check for training completion/errors
                if latest_log and latest_log.exists():
                    with open(latest_log, 'r') as f:
                        content = f.read()
                        if "Training completed" in content:
                            print("[SUCCESS] Training completed successfully!")
                            break
                        elif "Error" in content or "Exception" in content:
                            print("[WARNING] Training error detected - check logs above")
                            
                time.sleep(2)  # Update every 2 seconds
                
            except KeyboardInterrupt:
                print("\n[STOP] Monitoring stopped by user")
                break
            except Exception as e:
                print(f"[ERROR] Monitor error: {e}")
                time.sleep(5)
                
        self.running = False
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Mask R-CNN Training')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--config', type=str, default='food_7class', help='Config name to monitor')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_dir, args.config)
    
    try:
        monitor.monitor_logs()
    except KeyboardInterrupt:
        print("\n[EXIT] Exiting monitor...")
        monitor.stop()
        
if __name__ == "__main__":
    main()