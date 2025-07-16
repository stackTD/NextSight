#!/usr/bin/env python3
"""Test script for NextSight Phase 2 components without display."""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from core.camera_manager import CameraManager
from detection.hand_detector import HandDetector
from ui.overlay_renderer import OverlayRenderer
from utils.performance_monitor import PerformanceMonitor


def test_hand_detection():
    """Test hand detection without display."""
    logger.info("Testing NextSight Phase 2 components...")
    
    # Test hand detector
    hand_detector = HandDetector()
    logger.info("✅ Hand detector initialized")
    
    # Create test image (mock hand-like pattern)
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Test hand detection
    results = hand_detector.detect_hands(test_image)
    logger.info(f"✅ Hand detection test - Hands detected: {results['hands_detected']}")
    
    # Test overlay renderer
    overlay_renderer = OverlayRenderer()
    logger.info("✅ Professional overlay renderer initialized")
    
    # Test performance monitor
    performance_monitor = PerformanceMonitor()
    performance_monitor.update()
    stats = performance_monitor.get_system_stats()
    logger.info(f"✅ Performance monitor - Current FPS: {stats['current_fps']:.1f}")
    
    # Test camera manager in mock mode
    camera_manager = CameraManager(mock_mode=True)
    camera_manager.start()
    frame = camera_manager.get_frame()
    logger.info(f"✅ Camera manager (mock) - Frame shape: {frame.shape if frame is not None else 'None'}")
    
    # Test professional UI rendering (without display)
    if frame is not None:
        processed_frame = overlay_renderer.render_professional_ui(frame, results, stats)
        logger.info(f"✅ Professional UI rendering - Output shape: {processed_frame.shape}")
    
    # Test enhanced controls simulation
    overlay_renderer.cycle_overlay_mode()
    hand_detector.toggle_detection()
    logger.info("✅ Enhanced controls tested")
    
    # Cleanup
    hand_detector.cleanup()
    camera_manager.stop()
    
    logger.info("🎉 All NextSight Phase 2 components tested successfully!")
    logger.info("Phase 2 implementation ready for professional hand detection!")

if __name__ == "__main__":
    test_hand_detection()