# RTX 3060 Training Monitor GUI

A real-time monitoring dashboard for RTX 3060 food segmentation training with live graphs, GPU monitoring, and log analysis.

## Features

- **Real-time Training Progress**: Live loss graphs and epoch tracking
- **GPU Monitoring**: RTX 3060 utilization, temperature, memory, and power
- **Log File Browser**: Select and monitor any training log file
- **Live Updates**: Automatic refresh every 2 seconds
- **Dark Theme**: Professional dark interface optimized for monitoring

## Structure

```
gui_monitor/
├── main_dashboard.py      # Main GUI application
├── utils/
│   ├── __init__.py
│   ├── log_parser.py      # Training log parsing utility
│   └── gpu_monitor.py     # GPU statistics monitoring
├── scripts/               # Additional scripts (future use)
├── assets/               # GUI assets (future use)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### Quick Start
```bash
# Navigate to GUI folder
cd gui_monitor

# Install dependencies (if needed)
pip install -r requirements.txt

# Launch dashboard
python main_dashboard.py
```

### Features Overview

1. **Training Progress Tab**
   - Real-time loss graphs (training & validation)
   - Current epoch and batch progress
   - Training status indicators
   - Progress over time visualization

2. **GPU Status Tab**
   - RTX 3060 utilization graphs
   - Temperature monitoring
   - Memory usage (12GB VRAM tracking)
   - Power consumption monitoring

3. **Details Tab**
   - Live log file output
   - Recent training messages
   - Error detection and highlighting

### Log File Selection

The GUI automatically starts monitoring the current training log:
`logs/rtx3060_segmentation/training_20250831_164855.log`

You can:
- Browse for different log files
- Monitor any training session
- Switch between different experiments
- Real-time log tailing

### GPU Monitoring

Monitors your RTX 3060:
- **Utilization**: Real-time GPU usage percentage
- **Temperature**: Thermal monitoring with alerts
- **Memory**: 12GB VRAM usage tracking  
- **Power**: Power draw monitoring (up to 170W)

## Requirements

- Python 3.8+
- NVIDIA RTX 3060 with drivers 581.15+
- matplotlib, numpy, tkinter, Pillow
- nvidia-smi (included with drivers)

## Performance

The GUI is lightweight and updates every 2 seconds without impacting training performance.

## Troubleshooting

- **No GPU detected**: Ensure NVIDIA drivers are installed
- **Log file errors**: Check file permissions and path
- **Graph not updating**: Verify log file is being written to
- **Memory usage**: GUI uses minimal system resources (~50MB RAM)