#!/usr/bin/env python3
"""Performance test for NextSight Phase 4 bottle detection system."""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from core.bottle_controls import BottleControls


def create_test_frame():
    """Create a test frame for performance testing."""
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Add some bottle-like shapes for more realistic testing
    bottle_regions = [
        (200, 150, 120, 300),
        (400, 200, 100, 250), 
        (650, 100, 110, 350),
        (900, 180, 130, 280),
    ]
    
    for x, y, w, h in bottle_regions:
        color = (np.random.randint(100, 200), np.random.randint(100, 200), np.random.randint(100, 200))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    
    return frame


def test_detection_performance():
    """Test bottle detection performance."""
    logger.info("Starting NextSight Phase 4 performance test...")
    
    # Initialize bottle controls
    bottle_controls = BottleControls()
    
    if not bottle_controls.initialize():
        logger.error("Failed to initialize bottle controls")
        return False
    
    # Enable detection
    bottle_controls._toggle_bottle_detection()
    
    logger.info("Running performance test with 100 frames...")
    
    frame_times = []
    total_processing_times = []
    
    for i in range(100):
        # Create test frame
        test_frame = create_test_frame()
        
        # Measure frame processing time
        start_time = time.time()
        
        # Process bottle detection
        detection_results, detection_stats = bottle_controls.process_bottle_detection(test_frame)
        
        # Render UI (this is what would happen in real application)
        rendered_frame = bottle_controls.render_bottle_ui(test_frame, detection_results, detection_stats)
        
        end_time = time.time()
        
        frame_time = end_time - start_time
        frame_times.append(frame_time)
        
        if detection_results:
            total_processing_times.append(detection_results.processing_time)
        
        # Log progress every 20 frames
        if (i + 1) % 20 == 0:
            avg_time = np.mean(frame_times[-20:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            logger.info(f"Frame {i+1}/100 - Avg time: {avg_time*1000:.1f}ms, FPS: {fps:.1f}")
    
    # Calculate final statistics
    avg_frame_time = np.mean(frame_times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    min_frame_time = np.min(frame_times)
    max_frame_time = np.max(frame_times)
    max_fps = 1.0 / min_frame_time if min_frame_time > 0 else 0
    min_fps = 1.0 / max_frame_time if max_frame_time > 0 else 0
    
    avg_detection_time = np.mean(total_processing_times) if total_processing_times else 0
    
    logger.info("Performance Test Results:")
    logger.info(f"   Average FPS: {avg_fps:.1f}")
    logger.info(f"   Target FPS: 18-22")
    logger.info(f"   FPS Range: {min_fps:.1f} - {max_fps:.1f}")
    logger.info(f"   Average frame time: {avg_frame_time*1000:.1f}ms")
    logger.info(f"   Average detection time: {avg_detection_time*1000:.1f}ms")
    logger.info(f"   Target detection latency: <150ms")
    
    # Check if we meet performance targets
    meets_fps_target = 18 <= avg_fps <= 25  # Allow some margin above 22
    meets_latency_target = avg_detection_time < 0.15  # 150ms
    
    if meets_fps_target:
        logger.info("✅ FPS target met!")
    else:
        logger.warning(f"⚠️  FPS target not met. Target: 18-22, Actual: {avg_fps:.1f}")
    
    if meets_latency_target:
        logger.info("✅ Latency target met!")
    else:
        logger.warning(f"⚠️  Latency target not met. Target: <150ms, Actual: {avg_detection_time*1000:.1f}ms")
    
    # Test with different sensitivity levels
    logger.info("Testing performance with different sensitivity levels...")
    
    for sensitivity in ['low', 'medium', 'high']:
        bottle_controls.sensitivity_level = sensitivity
        bottle_controls.bottle_detector.adjust_sensitivity(sensitivity)
        
        # Run 20 frames for each sensitivity
        sensitivity_times = []
        for i in range(20):
            test_frame = create_test_frame()
            start_time = time.time()
            detection_results, detection_stats = bottle_controls.process_bottle_detection(test_frame)
            end_time = time.time()
            sensitivity_times.append(end_time - start_time)
        
        avg_time = np.mean(sensitivity_times)
        avg_fps_sensitivity = 1.0 / avg_time if avg_time > 0 else 0
        
        logger.info(f"   {sensitivity.upper()} sensitivity: {avg_fps_sensitivity:.1f} FPS ({avg_time*1000:.1f}ms)")
    
    # Cleanup
    bottle_controls.cleanup()
    
    overall_success = meets_fps_target and meets_latency_target
    
    if overall_success:
        logger.info("🎉 Performance test PASSED!")
    else:
        logger.warning("⚠️  Performance test needs optimization")
    
    return overall_success


if __name__ == "__main__":
    logger.info("Starting NextSight Phase 4 performance test...")
    
    if test_detection_performance():
        logger.info("✅ Performance test completed successfully")
    else:
        logger.warning("⚠️  Performance test completed with warnings")
    
    logger.info("Performance testing complete!")