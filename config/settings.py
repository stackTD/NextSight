"""Settings and configuration for NextSight."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Hardware-specific settings for Acer Nitro V, RTX 4050, Intel i5-13420H
HARDWARE_INFO = {
    "model": "Acer Nitro V",
    "gpu": "RTX 4050", 
    "cpu": "Intel i5-13420H",
    "optimized": True
}

# Camera settings optimized for RTX 4050
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1  # Minimal buffer for low latency
CAMERA_FORMAT = "MJPG"  # Better compression for RTX 4050
CAMERA_AUTO_EXPOSURE = True  # Handle varying lighting conditions
CAMERA_FOURCC = None  # Will be set programmatically

# Hand detection settings optimized for MediaPipe on RTX 4050
HAND_DETECTION_CONFIDENCE = 0.75  # Optimized for RTX 4050 performance
HAND_TRACKING_CONFIDENCE = 0.6   # Balanced for smooth tracking
MAX_NUM_HANDS = 2
HAND_MODEL_COMPLEXITY = 1  # Optimized for performance

# Object detection settings
OBJECT_DETECTION_CONFIDENCE = 0.6
JAR_CLASSIFICATION_THRESHOLD = 0.8

# Performance settings optimized for RTX 4050 and i5-13420H
ENABLE_GPU = True
GPU_MEMORY_GROWTH = True
USE_GPU_ACCELERATION = True
MAX_THREADS = 8  # i5-13420H has 8 cores
ENABLE_THREADING = True
FRAME_PROCESSING_THREADS = 2

# Display settings optimized for demo experience
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
DISPLAY_FPS = True
OVERLAY_ALPHA = 0.7
MIRROR_DISPLAY = True  # Better demo experience
WINDOW_NAME = "NextSight"
TARGET_FPS = 30
MAX_LATENCY_MS = 50  # Target latency camera to display

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
LOG_FILE = PROJECT_ROOT / "nextsight.log"
PERFORMANCE_LOG_INTERVAL = 30  # Log performance every 30 seconds

# Model paths
HAND_MODEL_PATH = MODELS_DIR / "hand_detection.tflite"
JAR_MODEL_PATH = MODELS_DIR / "jar_classifier.h5"

# Error handling settings
MAX_FRAME_FAILURES = 5  # Max consecutive frame read failures before restart
CAMERA_RECONNECT_ATTEMPTS = 3
CAMERA_RECONNECT_DELAY = 1.0  # seconds