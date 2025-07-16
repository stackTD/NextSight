"""Test camera manager functionality."""

import unittest
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from core.camera_manager import CameraManager


class TestCameraManager(unittest.TestCase):
    """Test cases for CameraManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_manager = CameraManager(mock_mode=True)
    
    def test_camera_initialization(self):
        """Test camera initialization in mock mode."""
        self.camera_manager.start()
        self.assertTrue(self.camera_manager.is_running)
        self.assertTrue(self.camera_manager.is_camera_healthy())
        
    def test_camera_info(self):
        """Test camera info retrieval."""
        self.camera_manager.start()
        info = self.camera_manager.get_camera_info()
        
        self.assertIsNotNone(info)
        self.assertEqual(info["mode"], "mock")
        self.assertEqual(info["width"], 1280)
        self.assertEqual(info["height"], 720)
        self.assertEqual(info["fps"], 30)
        
    def test_frame_capture(self):
        """Test frame capture functionality."""
        self.camera_manager.start()
        
        frame = self.camera_manager.get_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)  # Should be 3D (H, W, C)
        self.assertEqual(frame.shape[2], 3)    # Should have 3 channels
        self.assertEqual(frame.shape[0], 720)  # Height
        self.assertEqual(frame.shape[1], 1280) # Width
        
    def test_multiple_frames(self):
        """Test capturing multiple frames."""
        self.camera_manager.start()
        
        frames = []
        for i in range(5):
            frame = self.camera_manager.get_frame()
            self.assertIsNotNone(frame)
            frames.append(frame)
            time.sleep(0.1)
        
        self.assertEqual(len(frames), 5)
        # Frames should be different (mock camera generates different frames)
        self.assertFalse((frames[0] == frames[-1]).all())
        
    def test_camera_stop(self):
        """Test camera stop functionality."""
        self.camera_manager.start()
        self.assertTrue(self.camera_manager.is_running)
        
        self.camera_manager.stop()
        self.assertFalse(self.camera_manager.is_running)
        
        # Should return None after stop
        frame = self.camera_manager.get_frame()
        self.assertIsNone(frame)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.camera_manager.is_running:
            self.camera_manager.stop()


if __name__ == "__main__":
    unittest.main()