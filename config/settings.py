"""Settings and configuration for NextSight."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Hand detection settings
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Object detection settings
OBJECT_DETECTION_CONFIDENCE = 0.6
JAR_CLASSIFICATION_THRESHOLD = 0.8

# Performance settings
ENABLE_GPU = True
GPU_MEMORY_GROWTH = True

# Display settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
DISPLAY_FPS = True
OVERLAY_ALPHA = 0.7

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
LOG_FILE = PROJECT_ROOT / "nextsight.log"

# Model paths
HAND_MODEL_PATH = MODELS_DIR / "hand_detection.tflite"
JAR_MODEL_PATH = MODELS_DIR / "jar_classifier.h5"