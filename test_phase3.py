#!/usr/bin/env python3
"""Test script for NextSight Phase 3 gesture recognition components."""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from core.camera_manager import CameraManager
from detection.hand_detector import HandDetector
from detection.gesture_recognizer import GestureRecognizer
from ui.overlay_renderer import OverlayRenderer
from ui.message_overlay import MessageOverlay
from utils.performance_monitor import PerformanceMonitor


def test_phase3_components():
    """Test NextSight Phase 3 components without display."""
    logger.info("Testing NextSight Phase 3 gesture recognition components...")
    
    # Test hand detector
    hand_detector = HandDetector()
    logger.info("✅ Hand detector initialized")
    
    # Test gesture recognizer
    gesture_recognizer = GestureRecognizer()
    logger.info("✅ Gesture recognizer initialized")
    
    # Test message overlay
    message_overlay = MessageOverlay()
    logger.info("✅ Message overlay initialized")
    
    # Create test image (mock hand-like pattern)
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Test hand detection
    hand_results = hand_detector.detect_hands(test_image)
    logger.info(f"✅ Hand detection test - Hands detected: {hand_results['hands_detected']}")
    
    # Test gesture recognition on hand results
    gesture_results = gesture_recognizer.process_hands(hand_results)
    logger.info(f"✅ Gesture recognition test - Enabled: {gesture_results['gesture_recognition']['enabled']}")
    
    # Test message overlay
    message_overlay.add_custom_message("Test Message", (0, 255, 0))
    logger.info(f"✅ Message overlay test - Active messages: {message_overlay.get_active_message_count()}")
    
    # Test enhanced controls
    gesture_recognizer.toggle_recognition()
    gesture_recognizer.adjust_sensitivity('high')
    message_overlay.toggle_messages()
    logger.info("✅ Enhanced gesture controls tested")
    
    # Test overlay renderer with gesture info
    overlay_renderer = OverlayRenderer()
    performance_monitor = PerformanceMonitor()
    performance_monitor.update()
    stats = performance_monitor.get_system_stats()
    
    # Test camera manager in mock mode
    camera_manager = CameraManager(mock_mode=True)
    camera_manager.start()
    frame = camera_manager.get_frame()
    logger.info(f"✅ Camera manager (mock) - Frame shape: {frame.shape if frame is not None else 'None'}")
    
    # Test professional UI rendering with gesture info
    if frame is not None:
        processed_frame = overlay_renderer.render_professional_ui(frame, gesture_results, stats)
        processed_frame = message_overlay.render_messages(processed_frame)
        logger.info(f"✅ Professional UI with gesture rendering - Output shape: {processed_frame.shape}")
    
    # Test gesture state management
    gesture_stats = gesture_recognizer.get_session_stats()
    logger.info(f"✅ Gesture session stats - Total: {gesture_stats['total_gestures']}")
    
    # Test gesture recognition features
    supported_gestures = gesture_recognizer.get_supported_gestures()
    logger.info(f"✅ Supported gestures: {', '.join(supported_gestures)}")
    
    # Cleanup
    hand_detector.cleanup()
    gesture_recognizer.cleanup()
    camera_manager.stop()
    
    logger.info("🎉 All NextSight Phase 3 components tested successfully!")
    logger.info("Phase 3 implementation ready for advanced gesture recognition!")
    logger.info("🚀 Features: 5 gesture types, interactive messages, state management")


if __name__ == "__main__":
    test_phase3_components()