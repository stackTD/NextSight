# NextSight Setup Guide

This guide will walk you through setting up NextSight on your system.

## Prerequisites

### Hardware Requirements
- **Primary System**: Acer Nitro V with NVIDIA RTX 4050
- **Camera**: USB webcam or built-in camera
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB+ free space

### Software Requirements
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/stackTD/NextSight.git
cd NextSight
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install NextSight in development mode
pip install -e .
```

### 4. Configure GPU Support (RTX 4050)
```bash
# Install CUDA-enabled TensorFlow (if not already installed)
pip install tensorflow[and-cuda]

# Verify GPU detection
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### 5. Test Installation
```bash
# Run basic tests
python -m pytest tests/

# Test the main application
python src/main.py
```

## Configuration

### Camera Setup
1. Connect your camera
2. Update camera settings in `config/settings.py`:
   ```python
   CAMERA_ID = 0  # Change if using external camera
   CAMERA_WIDTH = 1280
   CAMERA_HEIGHT = 720
   ```

### Model Configuration
1. Place trained models in `data/models/`
2. Update model paths in `config/model_config.yaml`

## Troubleshooting

### Common Issues

**GPU Not Detected**
- Ensure NVIDIA drivers are installed
- Verify CUDA compatibility
- Check TensorFlow GPU installation

**Camera Access Issues**
- Check camera permissions
- Try different camera IDs (0, 1, 2...)
- Ensure camera is not being used by other applications

**Import Errors**
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Getting Help
- Check the [User Manual](user_manual.md)
- Review [API Reference](api_reference.md)
- Open an issue on GitHub

## Next Steps
Once installation is complete, refer to the [User Manual](user_manual.md) for usage instructions.