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
WINDOW_NAME = "NextSight - Professional Hand Detection"
TARGET_FPS = 30
MAX_LATENCY_MS = 50  # Target latency camera to display

# Professional UI settings for Phase 2
UI_THEME = {
    'background': (30, 30, 30),        # Dark background #1e1e1e
    'accent': (212, 120, 0),           # Accent blue #0078d4 (BGR)
    'success': (16, 124, 16),          # Success green #107c10 (BGR)
    'warning': (0, 165, 255),          # Warning orange
    'error': (0, 0, 255),              # Error red
    'text_primary': (255, 255, 255),   # White text
    'text_secondary': (200, 200, 200), # Light gray text
    'border': (100, 100, 100),         # Border gray
}

# UI Layout settings
UI_LAYOUT = {
    'title_height': 60,
    'status_panel_width': 300,
    'bottom_bar_height': 80,
    'margin': 10,
    'text_scale': 0.6,
    'title_scale': 0.8,
    'header_scale': 0.7,
}

# Hand landmark colors (BGR format)
HAND_COLORS = {
    'right_hand': (0, 255, 0),     # Green for right hand
    'left_hand': (255, 0, 0),      # Blue for left hand  
    'fingertips': (0, 0, 255),     # Red for fingertips
    'connections': (255, 255, 255), # White for connections
}

# Overlay display modes
OVERLAY_MODES = ['full', 'minimal', 'off']
DEFAULT_OVERLAY_MODE = 'full'

# Performance display settings
SHOW_PERFORMANCE_METRICS = True
SHOW_HAND_STATUS = True
SHOW_FINGER_COUNT = True
SHOW_CONFIDENCE = True

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
LOG_FILE = PROJECT_ROOT / "nextsight.log"
PERFORMANCE_LOG_INTERVAL = 30  # Log performance every 30 seconds

# Model paths
HAND_MODEL_PATH = MODELS_DIR / "hand_detection.tflite"
JAR_MODEL_PATH = MODELS_DIR / "jar_classifier.h5"

# Phase 3: Gesture Recognition Settings
GESTURE_RECOGNITION_ENABLED = True
GESTURE_CONFIDENCE_THRESHOLD = 0.8  # High accuracy requirement
GESTURE_HOLD_TIME = 0.5  # Minimum seconds to hold gesture
GESTURE_COOLDOWN_TIME = 2.0  # Cooldown between same gesture detections
GESTURE_FRAME_AVERAGING = 5  # Frames to average for stability
MAX_SIMULTANEOUS_GESTURES = 2  # One per hand
GESTURE_MESSAGE_DURATION = 3.0  # Message display time in seconds
GESTURE_ANIMATION_DURATION = 0.5  # Fade in/out time in seconds

# Gesture message configuration
GESTURE_MESSAGES = {
    'peace': {'text': 'Peace & Harmony! ✌️', 'color': (212, 120, 0)},      # Blue #0078d4 (BGR)
    'thumbs_up': {'text': 'Great Job! 👍', 'color': (16, 124, 16)},        # Green #107c10 (BGR)
    'thumbs_down': {'text': 'Not Good! 👎', 'color': (35, 17, 232)},       # Red #e81123 (BGR)
    'ok': {'text': 'Perfect! 👌', 'color': (0, 185, 255)},                # Gold #ffb900 (BGR)
    'stop': {'text': 'Detection Paused ⏸️', 'color': (0, 140, 255)}       # Orange #ff8c00 (BGR)
}

# Gesture detection thresholds and parameters
GESTURE_DETECTION_PARAMS = {
    'peace': {'tip_distance_threshold': 0.08, 'angle_threshold': 15.0},
    'thumbs_up': {'thumb_angle_threshold': 45.0, 'finger_curl_threshold': 0.7},
    'thumbs_down': {'thumb_angle_threshold': 45.0, 'finger_curl_threshold': 0.7},
    'ok': {'circle_distance_threshold': 0.05, 'finger_extension_threshold': 0.6},
    'stop': {'finger_extension_threshold': 0.8, 'spread_angle_threshold': 20.0}
}

# Error handling settings
MAX_FRAME_FAILURES = 5  # Max consecutive frame read failures before restart
CAMERA_RECONNECT_ATTEMPTS = 3
CAMERA_RECONNECT_DELAY = 1.0  # seconds