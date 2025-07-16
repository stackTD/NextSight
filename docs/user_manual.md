# NextSight User Manual

Welcome to NextSight! This manual will guide you through using the smart vision system.

## Overview

NextSight is an intelligent computer vision system that provides:
- Real-time hand tracking with finger detection
- Object detection and classification (jars with/without lids)
- Live camera feed with overlay visualizations
- Performance monitoring and statistics

## Getting Started

### Launch Application
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run NextSight
python src/main.py
```

### Main Interface

When you start NextSight, you'll see:
- Live camera feed in the main window
- Real-time hand tracking overlays
- Object detection bounding boxes
- FPS counter and performance metrics

### Controls
- **Q** or **ESC**: Quit the application
- **Space**: Pause/resume processing
- **S**: Save current frame
- **R**: Reset detection

## Features

### Hand Tracking
- Detects up to 2 hands simultaneously
- Shows 21 landmarks per hand
- Real-time finger tracking
- Gesture recognition (coming soon)

### Object Detection
- Identifies jars in the camera view
- Classifies jars as:
  - **OK**: Jar with lid (green box)
  - **NG**: Jar without lid (red box)
- Shows confidence scores

### Performance Monitoring
- Real-time FPS display
- CPU and memory usage
- Processing time statistics

## Configuration

### Adjusting Settings

Edit `config/settings.py` for basic settings:
```python
# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Detection thresholds
HAND_DETECTION_CONFIDENCE = 0.7
OBJECT_DETECTION_CONFIDENCE = 0.6
```

Edit `config/model_config.yaml` for advanced settings:
```yaml
hand_detection:
  confidence_threshold: 0.7
  max_hands: 2

object_detection:
  confidence_threshold: 0.6
  classification_threshold: 0.8
```

## Tips for Best Results

### Lighting
- Use good, even lighting
- Avoid backlighting
- Reduce shadows when possible

### Camera Position
- Position camera at eye level
- Ensure stable mounting
- Keep lens clean

### Hand Detection
- Keep hands visible and unobstructed
- Avoid rapid movements for better tracking
- Use contrasting backgrounds

### Object Detection
- Place objects clearly in frame
- Ensure good contrast with background
- Avoid overlapping objects

## Troubleshooting

### Poor Detection Performance
1. Check lighting conditions
2. Adjust confidence thresholds
3. Clean camera lens
4. Reduce background clutter

### Slow Performance
1. Check GPU utilization
2. Reduce camera resolution
3. Lower processing threads
4. Close other applications

### Application Crashes
1. Check log files in project root
2. Verify all dependencies are installed
3. Test with different camera settings

## Advanced Usage

### Training Custom Models
```bash
# Prepare training data
python scripts/data_preparation.py

# Train jar classifier
python scripts/train_model.py

# Export optimized model
python scripts/export_model.py
```

### Performance Benchmarking
```bash
# Run performance tests
python scripts/benchmark.py
```

## Support

For additional help:
- Check the [Setup Guide](setup_guide.md)
- Review [API Reference](api_reference.md)
- Submit issues on GitHub