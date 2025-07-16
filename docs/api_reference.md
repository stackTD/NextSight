# NextSight API Reference

This document provides detailed information about NextSight's API and modules.

## Core Modules

### CameraManager

Manages camera capture and video stream.

```python
from src.core.camera_manager import CameraManager

# Initialize camera
camera = CameraManager(camera_id=0)
camera.start()

# Get frames
frame = camera.get_frame()

# Cleanup
camera.stop()
```

**Methods:**
- `start()`: Initialize camera capture
- `stop()`: Stop camera and cleanup
- `get_frame()`: Get current frame from camera

### HandDetector

Hand detection and tracking using MediaPipe.

```python
from src.core.hand_detector import HandDetector

# Initialize detector
detector = HandDetector()

# Detect hands
hands = detector.detect_hands(image)

# Draw landmarks
detector.draw_landmarks(image, landmarks)
```

**Methods:**
- `detect_hands(image)`: Detect hands in image
- `draw_landmarks(image, landmarks)`: Draw hand landmarks

### ObjectDetector

Object detection and classification.

```python
from src.core.object_detector import ObjectDetector

# Initialize detector
detector = ObjectDetector()

# Detect objects
objects = detector.detect_objects(image)

# Classify jar
classification, confidence = detector.classify_jar(image, bbox)
```

**Methods:**
- `detect_objects(image)`: Detect objects in image
- `classify_jar(image, bbox)`: Classify jar with/without lid

## Model Classes

### JarClassifier

Machine learning model for jar classification.

```python
from src.models.jar_classifier import JarClassifier

# Initialize classifier
classifier = JarClassifier()
classifier.load_model()

# Make prediction
prediction, confidence = classifier.predict(image)
```

**Methods:**
- `load_model()`: Load trained model
- `predict(image)`: Predict jar classification
- `preprocess_image(image)`: Preprocess image for model

### ModelLoader

Utility for loading different model formats.

```python
from src.models.model_loader import ModelLoader

# Load TensorFlow model
model = ModelLoader.load_tensorflow_model(path)

# Load TFLite model
interpreter = ModelLoader.load_tflite_model(path)

# Check if model exists
exists = ModelLoader.check_model_exists(path)
```

**Methods:**
- `load_tensorflow_model(path)`: Load Keras model
- `load_tflite_model(path)`: Load TFLite model
- `check_model_exists(path)`: Check model file existence

## Utility Classes

### ImageUtils

Image processing utilities.

```python
from src.utils.image_utils import ImageUtils

# Resize image
resized = ImageUtils.resize_image(image, width, height)

# Normalize image
normalized = ImageUtils.normalize_image(image)

# Draw bounding box
ImageUtils.draw_bbox(image, bbox, color)

# Add text
ImageUtils.put_text(image, text, position)
```

**Methods:**
- `resize_image(image, width, height)`: Resize image
- `normalize_image(image)`: Normalize pixel values
- `crop_image(image, bbox)`: Crop image region
- `draw_bbox(image, bbox, color, thickness)`: Draw bounding box
- `put_text(image, text, position, color, font_scale)`: Add text

### PerformanceMonitor

Performance monitoring and statistics.

```python
from src.utils.performance_monitor import PerformanceMonitor

# Initialize monitor
monitor = PerformanceMonitor()

# Update metrics
monitor.update()

# Get statistics
fps = monitor.get_fps()
stats = monitor.get_system_stats()
monitor.log_stats()
```

**Methods:**
- `update()`: Update performance metrics
- `get_fps()`: Get current FPS
- `get_system_stats()`: Get system resource usage
- `log_stats()`: Log performance statistics

## UI Classes

### DisplayManager

Display window management.

```python
from src.ui.display_manager import DisplayManager

# Initialize display
display = DisplayManager()
display.create_window()

# Show frame
display.show_frame(frame)

# Check for exit
should_exit = display.check_exit()

# Cleanup
display.cleanup()
```

**Methods:**
- `create_window()`: Create display window
- `show_frame(frame)`: Display frame
- `check_exit()`: Check if user wants to exit
- `cleanup()`: Clean up display resources
- `resize_window(width, height)`: Resize window

### OverlayRenderer

Overlay rendering for visualizations.

```python
from src.ui.overlay_renderer import OverlayRenderer

# Initialize renderer
renderer = OverlayRenderer()

# Draw overlays
renderer.draw_hand_landmarks(image, landmarks)
renderer.draw_object_bbox(image, bbox, label, confidence)
renderer.draw_fps(image, fps)
renderer.draw_status(image, status_text)
```

**Methods:**
- `draw_hand_landmarks(image, landmarks)`: Draw hand landmarks
- `draw_object_bbox(image, bbox, label, confidence)`: Draw object box
- `draw_fps(image, fps)`: Draw FPS counter
- `draw_status(image, status_text, position)`: Draw status text

## Configuration

### Settings

Access configuration through `config.settings`:

```python
from config.settings import *

# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection thresholds
HAND_DETECTION_CONFIDENCE = 0.7
OBJECT_DETECTION_CONFIDENCE = 0.6

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
```

### Model Configuration

YAML configuration in `config/model_config.yaml`:

```yaml
hand_detection:
  confidence_threshold: 0.7
  max_hands: 2

object_detection:
  confidence_threshold: 0.6
  input_size: [224, 224]
  classes:
    - "jar_with_lid"
    - "jar_without_lid"
```

## Error Handling

All modules use the `loguru` logger for error reporting:

```python
from loguru import logger

try:
    # Your code here
    pass
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

## Examples

### Basic Usage

```python
from src.core.camera_manager import CameraManager
from src.core.hand_detector import HandDetector
from src.ui.display_manager import DisplayManager

# Initialize components
camera = CameraManager()
hand_detector = HandDetector()
display = DisplayManager()

# Main loop
camera.start()
display.create_window()

while True:
    frame = camera.get_frame()
    if frame is not None:
        hands = hand_detector.detect_hands(frame)
        hand_detector.draw_landmarks(frame, hands)
        display.show_frame(frame)
    
    if display.check_exit():
        break

# Cleanup
camera.stop()
display.cleanup()
```

### Advanced Processing

```python
from src.core import CameraManager, HandDetector, ObjectDetector
from src.core.result_processor import ResultProcessor
from src.ui import DisplayManager, OverlayRenderer
from src.utils.performance_monitor import PerformanceMonitor

# Initialize all components
camera = CameraManager()
hand_detector = HandDetector()
object_detector = ObjectDetector()
result_processor = ResultProcessor()
display = DisplayManager()
overlay = OverlayRenderer()
monitor = PerformanceMonitor()

# Main processing loop
while True:
    # Get frame
    frame = camera.get_frame()
    
    # Detect hands and objects
    hands = hand_detector.detect_hands(frame)
    objects = object_detector.detect_objects(frame)
    
    # Process results
    hand_results = result_processor.process_hand_results(hands)
    object_results = result_processor.process_object_results(objects)
    combined = result_processor.combine_results(hand_results, object_results)
    
    # Render overlays
    overlay.draw_hand_landmarks(frame, hands)
    for obj in objects:
        overlay.draw_object_bbox(frame, obj['bbox'], obj['label'], obj['confidence'])
    
    # Update performance
    monitor.update()
    overlay.draw_fps(frame, monitor.get_fps())
    
    # Display
    display.show_frame(frame)
    
    if display.check_exit():
        break
```