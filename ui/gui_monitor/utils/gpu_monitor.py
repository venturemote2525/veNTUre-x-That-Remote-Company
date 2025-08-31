#!/usr/bin/env python3
"""
GPU Monitor Utility for Training Monitor
Real-time GPU statistics monitoring using nvidia-smi
"""
import subprocess
import re
import time
from typing import Dict, List, Optional
from datetime import datetime

class GPUMonitor:
    """Monitor GPU statistics in real-time."""
    
    def __init__(self):
        self.nvidia_smi_available = self._check_nvidia_smi()
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def get_gpu_stats(self) -> Optional[Dict]:
        """
        Get current GPU statistics.
        
        Returns:
            Dictionary containing GPU stats or None if not available
        """
        if not self.nvidia_smi_available:
            return None
        
        try:
            # Query GPU statistics
            cmd = [
                'nvidia-smi',
                '--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return None
            
            # Parse first GPU (index 0)
            data = lines[0].split(', ')
            
            if len(data) < 9:
                return None
            
            gpu_stats = {
                'timestamp': data[0].strip(),
                'name': data[1].strip(),
                'temperature': self._safe_float(data[2]),
                'utilization': self._safe_float(data[3]),
                'memory_utilization': self._safe_float(data[4]),
                'memory_used': self._safe_float(data[5]),
                'memory_total': self._safe_float(data[6]),
                'power_draw': self._safe_float(data[7]),
                'power_limit': self._safe_float(data[8])
            }
            
            return gpu_stats
            
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return None
    
    def get_gpu_processes(self) -> List[Dict]:
        """
        Get list of processes using GPU.
        
        Returns:
            List of dictionaries containing process information
        """
        if not self.nvidia_smi_available:
            return []
        
        try:
            cmd = [
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return []
            
            processes = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    data = line.split(', ')
                    if len(data) >= 3:
                        processes.append({
                            'pid': data[0].strip(),
                            'name': data[1].strip(),
                            'memory_used': self._safe_float(data[2])
                        })
            
            return processes
            
        except Exception as e:
            print(f"Error getting GPU processes: {e}")
            return []
    
    def get_detailed_info(self) -> Optional[Dict]:
        """
        Get detailed GPU information.
        
        Returns:
            Dictionary with detailed GPU info
        """
        if not self.nvidia_smi_available:
            return None
        
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=name,driver_version,memory.total,compute_cap,gpu_bus_id',
                '--format=csv,noheader'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return None
            
            data = lines[0].split(', ')
            
            if len(data) < 5:
                return None
            
            info = {
                'name': data[0].strip(),
                'driver_version': data[1].strip(),
                'memory_total': data[2].strip(),
                'compute_capability': data[3].strip(),
                'bus_id': data[4].strip()
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting detailed GPU info: {e}")
            return None
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            # Remove any non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d\.]', '', str(value))
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    def monitor_continuous(self, duration_seconds: int = 60, interval: float = 1.0) -> List[Dict]:
        """
        Monitor GPU continuously for a specified duration.
        
        Args:
            duration_seconds: How long to monitor
            interval: Interval between measurements in seconds
            
        Returns:
            List of GPU stat dictionaries over time
        """
        if not self.nvidia_smi_available:
            return []
        
        stats_history = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            stats = self.get_gpu_stats()
            if stats:
                stats['measurement_time'] = datetime.now()
                stats_history.append(stats)
            
            time.sleep(interval)
        
        return stats_history
    
    def get_training_performance_summary(self, stats_history: List[Dict]) -> Dict:
        """
        Generate performance summary from GPU stats history.
        
        Args:
            stats_history: List of GPU stats over time
            
        Returns:
            Dictionary with performance summary
        """
        if not stats_history:
            return {}
        
        utilizations = [s.get('utilization', 0) for s in stats_history]
        temperatures = [s.get('temperature', 0) for s in stats_history]
        power_draws = [s.get('power_draw', 0) for s in stats_history]
        memory_used = [s.get('memory_used', 0) for s in stats_history]
        
        summary = {
            'avg_utilization': sum(utilizations) / len(utilizations) if utilizations else 0,
            'max_utilization': max(utilizations) if utilizations else 0,
            'min_utilization': min(utilizations) if utilizations else 0,
            'avg_temperature': sum(temperatures) / len(temperatures) if temperatures else 0,
            'max_temperature': max(temperatures) if temperatures else 0,
            'avg_power_draw': sum(power_draws) / len(power_draws) if power_draws else 0,
            'max_power_draw': max(power_draws) if power_draws else 0,
            'avg_memory_used': sum(memory_used) / len(memory_used) if memory_used else 0,
            'max_memory_used': max(memory_used) if memory_used else 0,
            'measurement_count': len(stats_history),
            'duration': (stats_history[-1]['measurement_time'] - stats_history[0]['measurement_time']).total_seconds() if len(stats_history) > 1 else 0
        }
        
        return summary
    
    def is_training_active(self) -> bool:
        """
        Check if training is currently active based on GPU utilization.
        
        Returns:
            True if training appears to be active
        """
        stats = self.get_gpu_stats()
        if not stats:
            return False
        
        # Consider training active if GPU utilization > 10%
        return stats.get('utilization', 0) > 10
    
    def get_memory_info(self) -> Optional[Dict]:
        """
        Get detailed memory information.
        
        Returns:
            Dictionary with memory details
        """
        stats = self.get_gpu_stats()
        if not stats:
            return None
        
        memory_total = stats.get('memory_total', 0)
        memory_used = stats.get('memory_used', 0)
        memory_free = memory_total - memory_used
        memory_utilization = (memory_used / memory_total * 100) if memory_total > 0 else 0
        
        return {
            'total_mb': memory_total,
            'used_mb': memory_used,
            'free_mb': memory_free,
            'utilization_percent': memory_utilization,
            'total_gb': memory_total / 1024,
            'used_gb': memory_used / 1024,
            'free_gb': memory_free / 1024
        }