"""Tests for hand detection functionality."""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from core.hand_detector import HandDetector


class TestHandDetection(unittest.TestCase):
    """Test cases for hand detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HandDetector()
    
    def test_detector_initialization(self):
        """Test hand detector initialization."""
        self.assertIsNotNone(self.detector)
    
    def test_detect_hands_empty_image(self):
        """Test hand detection with empty image."""
        # Create a blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect_hands(image)
        self.assertIsInstance(results, list)
    
    def test_detect_hands_valid_image(self):
        """Test hand detection with valid image."""
        # Create a test image (placeholder)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.detector.detect_hands(image)
        self.assertIsInstance(results, list)
    
    def tearDown(self):
        """Clean up test fixtures."""
        pass


if __name__ == "__main__":
    unittest.main()