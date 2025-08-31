#!/usr/bin/env python3
"""
RTX 3060 Training Monitor - Main Dashboard GUI
Real-time monitoring of training progress, GPU status, and loss graphs
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from log_parser import LogParser
from gpu_monitor import GPUMonitor

class TrainingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RTX 3060 Training Monitor - Food Segmentation")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize monitors
        self.log_parser = LogParser()
        self.gpu_monitor = GPUMonitor()
        
        # Data storage
        self.training_data = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'timestamps': []
        }
        
        self.gpu_data = {
            'timestamps': [],
            'gpu_util': [],
            'gpu_temp': [],
            'gpu_memory': [],
            'gpu_power': []
        }
        
        # Current log file
        self.current_log_file = "logs/rtx3060_segmentation/training_20250831_164855.log"
        
        self.setup_ui()
        self.start_monitoring()
    
    def setup_ui(self):
        """Setup the main UI components."""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(title_frame, text="🚀 RTX 3060 Training Monitor", 
                             font=('Arial', 18, 'bold'), fg='#00ff00', bg='#2b2b2b')
        title_label.pack(side='left')
        
        # Status indicators
        self.status_frame = tk.Frame(title_frame, bg='#2b2b2b')
        self.status_frame.pack(side='right')
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#333333', relief='ridge', bd=2)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Log file selection
        log_frame = tk.Frame(control_frame, bg='#333333')
        log_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(log_frame, text="Log File:", fg='white', bg='#333333', font=('Arial', 10, 'bold')).pack(side='left')
        
        self.log_path_var = tk.StringVar(value=self.current_log_file)
        self.log_entry = tk.Entry(log_frame, textvariable=self.log_path_var, width=60, bg='#555555', fg='white')
        self.log_entry.pack(side='left', padx=5)
        
        tk.Button(log_frame, text="Browse", command=self.browse_log_file, 
                 bg='#0078d4', fg='white', font=('Arial', 9, 'bold')).pack(side='left', padx=2)
        
        tk.Button(log_frame, text="Refresh", command=self.refresh_data, 
                 bg='#107c10', fg='white', font=('Arial', 9, 'bold')).pack(side='left', padx=2)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Training tab
        self.setup_training_tab()
        
        # GPU tab
        self.setup_gpu_tab()
        
        # Details tab
        self.setup_details_tab()
    
    def setup_training_tab(self):
        """Setup training monitoring tab."""
        training_frame = tk.Frame(self.notebook, bg='#2b2b2b')
        self.notebook.add(training_frame, text="🎯 Training Progress")
        
        # Training info frame
        info_frame = tk.Frame(training_frame, bg='#333333', relief='ridge', bd=2)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        # Current training status
        self.training_status = tk.Label(info_frame, text="Status: Initializing...", 
                                      fg='#ffaa00', bg='#333333', font=('Arial', 12, 'bold'))
        self.training_status.pack(side='left', padx=10, pady=5)
        
        self.current_epoch = tk.Label(info_frame, text="Epoch: --/--", 
                                    fg='white', bg='#333333', font=('Arial', 10))
        self.current_epoch.pack(side='left', padx=20)
        
        self.current_loss = tk.Label(info_frame, text="Loss: --", 
                                   fg='white', bg='#333333', font=('Arial', 10))
        self.current_loss.pack(side='left', padx=20)
        
        # Training graphs
        graph_frame = tk.Frame(training_frame, bg='#2b2b2b')
        graph_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Loss graph
        self.fig_training, (self.ax_loss, self.ax_progress) = plt.subplots(2, 1, figsize=(12, 8), 
                                                                          facecolor='#2b2b2b')
        self.fig_training.suptitle('Training Progress', color='white', fontsize=14, fontweight='bold')
        
        # Loss plot
        self.ax_loss.set_facecolor('#333333')
        self.ax_loss.set_xlabel('Epoch', color='white')
        self.ax_loss.set_ylabel('Loss', color='white')
        self.ax_loss.set_title('Training & Validation Loss', color='white', fontweight='bold')
        self.ax_loss.tick_params(colors='white')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Progress plot
        self.ax_progress.set_facecolor('#333333')
        self.ax_progress.set_xlabel('Time', color='white')
        self.ax_progress.set_ylabel('Batch Progress', color='white')
        self.ax_progress.set_title('Training Progress Over Time', color='white', fontweight='bold')
        self.ax_progress.tick_params(colors='white')
        self.ax_progress.grid(True, alpha=0.3)
        
        self.canvas_training = FigureCanvasTkAgg(self.fig_training, graph_frame)
        self.canvas_training.draw()
        self.canvas_training.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_gpu_tab(self):
        """Setup GPU monitoring tab."""
        gpu_frame = tk.Frame(self.notebook, bg='#2b2b2b')
        self.notebook.add(gpu_frame, text="🔥 GPU Status")
        
        # GPU info frame
        gpu_info_frame = tk.Frame(gpu_frame, bg='#333333', relief='ridge', bd=2)
        gpu_info_frame.pack(fill='x', padx=5, pady=5)
        
        self.gpu_name = tk.Label(gpu_info_frame, text="GPU: RTX 3060", 
                               fg='#00ff00', bg='#333333', font=('Arial', 12, 'bold'))
        self.gpu_name.pack(side='left', padx=10, pady=5)
        
        self.gpu_util_label = tk.Label(gpu_info_frame, text="Utilization: --%", 
                                     fg='white', bg='#333333', font=('Arial', 10))
        self.gpu_util_label.pack(side='left', padx=20)
        
        self.gpu_temp_label = tk.Label(gpu_info_frame, text="Temperature: --°C", 
                                     fg='white', bg='#333333', font=('Arial', 10))
        self.gpu_temp_label.pack(side='left', padx=20)
        
        self.gpu_mem_label = tk.Label(gpu_info_frame, text="Memory: --/12GB", 
                                    fg='white', bg='#333333', font=('Arial', 10))
        self.gpu_mem_label.pack(side='left', padx=20)
        
        # GPU graphs
        gpu_graph_frame = tk.Frame(gpu_frame, bg='#2b2b2b')
        gpu_graph_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.fig_gpu, ((self.ax_util, self.ax_temp), (self.ax_mem, self.ax_power)) = plt.subplots(
            2, 2, figsize=(12, 8), facecolor='#2b2b2b')
        self.fig_gpu.suptitle('RTX 3060 GPU Monitoring', color='white', fontsize=14, fontweight='bold')
        
        # Setup GPU subplots
        gpu_axes = [
            (self.ax_util, 'GPU Utilization (%)', '#00ff00'),
            (self.ax_temp, 'GPU Temperature (°C)', '#ff6600'),
            (self.ax_mem, 'GPU Memory (GB)', '#0078d4'),
            (self.ax_power, 'GPU Power (W)', '#ff0066')
        ]
        
        for ax, title, color in gpu_axes:
            ax.set_facecolor('#333333')
            ax.set_title(title, color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
        
        self.canvas_gpu = FigureCanvasTkAgg(self.fig_gpu, gpu_graph_frame)
        self.canvas_gpu.draw()
        self.canvas_gpu.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_details_tab(self):
        """Setup detailed information tab."""
        details_frame = tk.Frame(self.notebook, bg='#2b2b2b')
        self.notebook.add(details_frame, text="📋 Details")
        
        # Log output
        log_frame = tk.Frame(details_frame, bg='#333333', relief='ridge', bd=2)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        tk.Label(log_frame, text="Recent Training Log Output:", 
                fg='white', bg='#333333', font=('Arial', 12, 'bold')).pack(anchor='w', padx=5, pady=5)
        
        # Text widget with scrollbar
        text_frame = tk.Frame(log_frame, bg='#333333')
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(text_frame, bg='#1a1a1a', fg='#00ff00', font=('Courier', 9))
        scrollbar = tk.Scrollbar(text_frame)
        
        scrollbar.pack(side='right', fill='y')
        self.log_text.pack(side='left', fill='both', expand=True)
        
        scrollbar.config(command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
    
    def browse_log_file(self):
        """Browse for log file."""
        filename = filedialog.askopenfilename(
            title="Select Training Log File",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.current_log_file = filename
            self.log_path_var.set(filename)
            self.refresh_data()
    
    def refresh_data(self):
        """Refresh training and GPU data."""
        try:
            # Update log file path
            self.current_log_file = self.log_path_var.get()
            
            # Parse training data
            if os.path.exists(self.current_log_file):
                training_info = self.log_parser.parse_log(self.current_log_file)
                self.update_training_data(training_info)
            
            # Get GPU data
            gpu_info = self.gpu_monitor.get_gpu_stats()
            self.update_gpu_data(gpu_info)
            
            # Update UI
            self.update_ui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh data: {str(e)}")
    
    def update_training_data(self, training_info):
        """Update training data from log."""
        if training_info:
            self.training_data = training_info
    
    def update_gpu_data(self, gpu_info):
        """Update GPU data."""
        if gpu_info:
            current_time = datetime.now()
            
            # Keep only recent data (last 100 points)
            max_points = 100
            
            self.gpu_data['timestamps'].append(current_time)
            self.gpu_data['gpu_util'].append(gpu_info.get('utilization', 0))
            self.gpu_data['gpu_temp'].append(gpu_info.get('temperature', 0))
            self.gpu_data['gpu_memory'].append(gpu_info.get('memory_used', 0) / 1024)  # Convert to GB
            self.gpu_data['gpu_power'].append(gpu_info.get('power_draw', 0))
            
            # Limit data points
            for key in self.gpu_data:
                if len(self.gpu_data[key]) > max_points:
                    self.gpu_data[key] = self.gpu_data[key][-max_points:]
    
    def update_ui(self):
        """Update UI elements with latest data."""
        # Update training status
        if self.training_data['epochs']:
            current_epoch = len(self.training_data['epochs'])
            self.training_status.config(text="Status: Training Active", fg='#00ff00')
            self.current_epoch.config(text=f"Epoch: {current_epoch}")
            
            if self.training_data['train_loss']:
                latest_loss = self.training_data['train_loss'][-1]
                self.current_loss.config(text=f"Loss: {latest_loss:.4f}")
        
        # Update GPU status
        if self.gpu_data['gpu_util']:
            latest_util = self.gpu_data['gpu_util'][-1]
            latest_temp = self.gpu_data['gpu_temp'][-1]
            latest_mem = self.gpu_data['gpu_memory'][-1]
            
            self.gpu_util_label.config(text=f"Utilization: {latest_util}%")
            self.gpu_temp_label.config(text=f"Temperature: {latest_temp}°C")
            self.gpu_mem_label.config(text=f"Memory: {latest_mem:.1f}/12GB")
        
        # Update graphs
        self.update_training_graphs()
        self.update_gpu_graphs()
        
        # Update log text
        self.update_log_display()
    
    def update_training_graphs(self):
        """Update training progress graphs."""
        if not self.training_data['epochs']:
            return
        
        # Clear axes
        self.ax_loss.clear()
        self.ax_progress.clear()
        
        # Plot loss
        epochs = self.training_data['epochs']
        if self.training_data['train_loss']:
            self.ax_loss.plot(epochs, self.training_data['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if self.training_data['val_loss']:
            self.ax_loss.plot(epochs, self.training_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        self.ax_loss.set_facecolor('#333333')
        self.ax_loss.set_xlabel('Epoch', color='white')
        self.ax_loss.set_ylabel('Loss', color='white')
        self.ax_loss.set_title('Training & Validation Loss', color='white', fontweight='bold')
        self.ax_loss.tick_params(colors='white')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend()
        
        # Plot progress over time
        if self.training_data['timestamps']:
            times = [(t - self.training_data['timestamps'][0]).total_seconds() / 3600 for t in self.training_data['timestamps']]
            self.ax_progress.plot(times, epochs, 'g-', linewidth=2, marker='o', markersize=4)
        
        self.ax_progress.set_facecolor('#333333')
        self.ax_progress.set_xlabel('Training Time (hours)', color='white')
        self.ax_progress.set_ylabel('Epoch', color='white')
        self.ax_progress.set_title('Training Progress Over Time', color='white', fontweight='bold')
        self.ax_progress.tick_params(colors='white')
        self.ax_progress.grid(True, alpha=0.3)
        
        self.canvas_training.draw()
    
    def update_gpu_graphs(self):
        """Update GPU monitoring graphs."""
        if not self.gpu_data['timestamps']:
            return
        
        # Time axis (last N minutes)
        times = [(t - self.gpu_data['timestamps'][0]).total_seconds() / 60 for t in self.gpu_data['timestamps']]
        
        # GPU Utilization
        self.ax_util.clear()
        self.ax_util.plot(times, self.gpu_data['gpu_util'], '#00ff00', linewidth=2)
        self.ax_util.set_facecolor('#333333')
        self.ax_util.set_title('GPU Utilization (%)', color='white', fontweight='bold')
        self.ax_util.set_ylabel('%', color='white')
        self.ax_util.tick_params(colors='white')
        self.ax_util.grid(True, alpha=0.3)
        self.ax_util.set_ylim(0, 100)
        
        # GPU Temperature
        self.ax_temp.clear()
        self.ax_temp.plot(times, self.gpu_data['gpu_temp'], '#ff6600', linewidth=2)
        self.ax_temp.set_facecolor('#333333')
        self.ax_temp.set_title('GPU Temperature (°C)', color='white', fontweight='bold')
        self.ax_temp.set_ylabel('°C', color='white')
        self.ax_temp.tick_params(colors='white')
        self.ax_temp.grid(True, alpha=0.3)
        
        # GPU Memory
        self.ax_mem.clear()
        self.ax_mem.plot(times, self.gpu_data['gpu_memory'], '#0078d4', linewidth=2)
        self.ax_mem.set_facecolor('#333333')
        self.ax_mem.set_title('GPU Memory (GB)', color='white', fontweight='bold')
        self.ax_mem.set_ylabel('GB', color='white')
        self.ax_mem.set_xlabel('Time (minutes)', color='white')
        self.ax_mem.tick_params(colors='white')
        self.ax_mem.grid(True, alpha=0.3)
        self.ax_mem.set_ylim(0, 12)
        
        # GPU Power
        self.ax_power.clear()
        self.ax_power.plot(times, self.gpu_data['gpu_power'], '#ff0066', linewidth=2)
        self.ax_power.set_facecolor('#333333')
        self.ax_power.set_title('GPU Power (W)', color='white', fontweight='bold')
        self.ax_power.set_ylabel('W', color='white')
        self.ax_power.set_xlabel('Time (minutes)', color='white')
        self.ax_power.tick_params(colors='white')
        self.ax_power.grid(True, alpha=0.3)
        
        self.canvas_gpu.draw()
    
    def update_log_display(self):
        """Update log text display."""
        try:
            if os.path.exists(self.current_log_file):
                with open(self.current_log_file, 'r') as f:
                    lines = f.readlines()
                    # Show last 50 lines
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(tk.END, ''.join(recent_lines))
                self.log_text.see(tk.END)  # Scroll to bottom
        except Exception as e:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"Error reading log file: {str(e)}")
    
    def start_monitoring(self):
        """Start the monitoring loop."""
        self.refresh_data()
        # Schedule next update
        self.root.after(2000, self.start_monitoring)  # Update every 2 seconds

def main():
    """Main function to start the dashboard."""
    root = tk.Tk()
    
    # Set dark theme style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configure styles
    style.configure('TNotebook', background='#2b2b2b', borderwidth=0)
    style.configure('TNotebook.Tab', background='#404040', foreground='white', 
                   padding=[12, 8], focuscolor='none')
    style.map('TNotebook.Tab', background=[('selected', '#0078d4'), ('active', '#005a9e')])
    
    app = TrainingDashboard(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Dashboard closed by user")

if __name__ == "__main__":
    main()