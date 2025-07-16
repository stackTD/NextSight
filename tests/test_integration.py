"""Integration tests for NextSight application."""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from core.camera_manager import CameraManager
from core.hand_detector import HandDetector
from core.object_detector import ObjectDetector
from core.result_processor import ResultProcessor
from ui.display_manager import DisplayManager
from ui.overlay_renderer import OverlayRenderer


class TestIntegration(unittest.TestCase):
    """Integration test cases for NextSight."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_manager = CameraManager()
        self.hand_detector = HandDetector()
        self.object_detector = ObjectDetector()
        self.result_processor = ResultProcessor()
        self.display_manager = DisplayManager()
        self.overlay_renderer = OverlayRenderer()
    
    def test_all_components_initialize(self):
        """Test that all components initialize without errors."""
        self.assertIsNotNone(self.camera_manager)
        self.assertIsNotNone(self.hand_detector)
        self.assertIsNotNone(self.object_detector)
        self.assertIsNotNone(self.result_processor)
        self.assertIsNotNone(self.display_manager)
        self.assertIsNotNone(self.overlay_renderer)
    
    def test_processing_pipeline(self):
        """Test the complete processing pipeline."""
        # Create a test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test hand detection
        hand_results = self.hand_detector.detect_hands(image)
        self.assertIsInstance(hand_results, list)
        
        # Test object detection
        object_results = self.object_detector.detect_objects(image)
        self.assertIsInstance(object_results, list)
        
        # Test result processing
        processed_hands = self.result_processor.process_hand_results(hand_results)
        processed_objects = self.result_processor.process_object_results(object_results)
        combined_results = self.result_processor.combine_results(processed_hands, processed_objects)
        
        self.assertIsInstance(processed_hands, dict)
        self.assertIsInstance(processed_objects, dict)
        self.assertIsInstance(combined_results, dict)
    
    def tearDown(self):
        """Clean up test fixtures."""
        pass


if __name__ == "__main__":
    unittest.main()