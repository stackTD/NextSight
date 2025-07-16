#!/usr/bin/env python3
"""Generate NextSight Phase 2 demo screenshots without display."""

import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from core.camera_manager import CameraManager
from detection.hand_detector import HandDetector
from ui.overlay_renderer import OverlayRenderer
from utils.performance_monitor import PerformanceMonitor


def create_demo_screenshots():
    """Create professional demo screenshots showing Phase 2 capabilities."""
    logger.info("Generating NextSight Phase 2 demo screenshots...")
    
    # Initialize components
    camera_manager = CameraManager(mock_mode=True)
    camera_manager.start()
    
    hand_detector = HandDetector()
    overlay_renderer = OverlayRenderer()
    performance_monitor = PerformanceMonitor()
    
    # Generate frames for different scenarios
    scenarios = [
        ("no_hands", "Professional UI - No Hands Detected"),
        ("minimal_mode", "Minimal Overlay Mode"),
        ("full_mode", "Full Professional UI Mode")
    ]
    
    for i, (scenario_name, description) in enumerate(scenarios):
        logger.info(f"Generating scenario: {description}")
        
        # Get mock frame
        frame = camera_manager.get_frame()
        if frame is None:
            continue
        
        # Update performance for realistic stats
        for _ in range(10):
            performance_monitor.update()
            time.sleep(0.001)  # Simulate processing time
        
        stats = performance_monitor.get_system_stats()
        
        # Create mock detection results for demonstration
        if scenario_name == "no_hands":
            detection_results = {
                'hands_detected': 0,
                'hands': [],
                'total_fingers': 0,
                'left_fingers': 0,
                'right_fingers': 0,
                'raw_results': None,
                'confidence_avg': 0.0
            }
            overlay_renderer.set_overlay_mode('full')
        elif scenario_name == "minimal_mode":
            detection_results = {
                'hands_detected': 0,
                'hands': [],
                'total_fingers': 0,
                'left_fingers': 0,
                'right_fingers': 0,
                'raw_results': None,
                'confidence_avg': 0.0
            }
            overlay_renderer.set_overlay_mode('minimal')
        else:  # full_mode with simulated hands
            # Create mock hand landmarks for demonstration
            detection_results = {
                'hands_detected': 2,
                'hands': [
                    {
                        'index': 0,
                        'label': 'Left',
                        'confidence': 0.95,
                        'landmarks': None,  # Would contain MediaPipe landmarks
                        'fingers_up': [1, 1, 1, 0, 0],
                        'finger_count': 3
                    },
                    {
                        'index': 1,
                        'label': 'Right',
                        'confidence': 0.88,
                        'landmarks': None,
                        'fingers_up': [1, 1, 1, 1, 1],
                        'finger_count': 5
                    }
                ],
                'total_fingers': 8,
                'left_fingers': 3,
                'right_fingers': 5,
                'raw_results': None,
                'confidence_avg': 0.915
            }
            overlay_renderer.set_overlay_mode('full')
        
        # Render professional UI
        demo_frame = overlay_renderer.render_professional_ui(frame, detection_results, stats)
        
        # Add demo watermark
        demo_text = f"NextSight Phase 2 Demo - {description}"
        cv2.putText(demo_frame, demo_text, (10, demo_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save screenshot
        filename = f"nextsight_phase2_demo_{scenario_name}.jpg"
        cv2.imwrite(filename, demo_frame)
        logger.info(f"✅ Saved demo screenshot: {filename}")
    
    # Generate one special screenshot showing the professional branding
    logger.info("Generating professional branding showcase...")
    frame = camera_manager.get_frame()
    stats = performance_monitor.get_system_stats()
    
    # Professional showcase with enhanced branding
    detection_results = {
        'hands_detected': 1,
        'hands': [
            {
                'index': 0,
                'label': 'Right',
                'confidence': 0.92,
                'landmarks': None,
                'fingers_up': [0, 1, 1, 0, 0],
                'finger_count': 2
            }
        ],
        'total_fingers': 2,
        'left_fingers': 0,
        'right_fingers': 2,
        'raw_results': None,
        'confidence_avg': 0.92
    }
    
    overlay_renderer.set_overlay_mode('full')
    showcase_frame = overlay_renderer.render_professional_ui(frame, detection_results, stats)
    
    # Add professional branding showcase text
    cv2.putText(showcase_frame, "NextSight Phase 2: Professional Hand Detection System", 
               (10, showcase_frame.shape[0] - 40), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (212, 120, 0), 2)
    cv2.putText(showcase_frame, "RTX 4050 Optimized | Real-time Performance | Advanced UI", 
               (10, showcase_frame.shape[0] - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite("nextsight_phase2_professional_showcase.jpg", showcase_frame)
    logger.info("✅ Saved professional showcase: nextsight_phase2_professional_showcase.jpg")
    
    # Cleanup
    hand_detector.cleanup()
    camera_manager.stop()
    
    logger.info("🎉 All demo screenshots generated successfully!")
    logger.info("Screenshots ready for professional demonstration:")
    logger.info("  - nextsight_phase2_demo_no_hands.jpg")
    logger.info("  - nextsight_phase2_demo_minimal_mode.jpg") 
    logger.info("  - nextsight_phase2_demo_full_mode.jpg")
    logger.info("  - nextsight_phase2_professional_showcase.jpg")


if __name__ == "__main__":
    create_demo_screenshots()