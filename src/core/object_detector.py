"""Object detection and classification."""

from loguru import logger
from config.settings import OBJECT_DETECTION_CONFIDENCE


class ObjectDetector:
    """Object detection and classification for jars."""
    
    def __init__(self):
        """Initialize object detector."""
        # TODO: Load object detection model
        logger.info("Object detector initialized")
    
    def detect_objects(self, image):
        """Detect objects in the given image."""
        # TODO: Implement object detection
        return []
    
    def classify_jar(self, image, bbox):
        """Classify if jar has lid or not."""
        # TODO: Implement jar classification
        return "unknown", 0.0