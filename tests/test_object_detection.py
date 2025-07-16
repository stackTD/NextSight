"""Tests for object detection functionality."""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from core.object_detector import ObjectDetector


class TestObjectDetection(unittest.TestCase):
    """Test cases for object detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ObjectDetector()
    
    def test_detector_initialization(self):
        """Test object detector initialization."""
        self.assertIsNotNone(self.detector)
    
    def test_detect_objects_empty_image(self):
        """Test object detection with empty image."""
        # Create a blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect_objects(image)
        self.assertIsInstance(results, list)
    
    def test_classify_jar(self):
        """Test jar classification."""
        # Create a test image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        bbox = (0, 0, 224, 224)
        classification, confidence = self.detector.classify_jar(image, bbox)
        self.assertIsInstance(classification, str)
        self.assertIsInstance(confidence, (int, float))
    
    def tearDown(self):
        """Clean up test fixtures."""
        pass


if __name__ == "__main__":
    unittest.main()