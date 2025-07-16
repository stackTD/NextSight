#!/usr/bin/env python3
"""Performance verification test for NextSight Phase 3."""

import sys
import time
import psutil
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from detection.hand_detector import HandDetector
from detection.gesture_recognizer import GestureRecognizer
from ui.message_overlay import MessageOverlay
from ui.overlay_renderer import OverlayRenderer
from utils.performance_monitor import PerformanceMonitor


def test_phase3_performance_targets():
    """Verify Phase 3 meets all performance requirements."""
    logger.info("🎯 Testing NextSight Phase 3 Performance Targets...")
    
    # Initialize all components
    hand_detector = HandDetector()
    gesture_recognizer = GestureRecognizer()
    message_overlay = MessageOverlay()
    overlay_renderer = OverlayRenderer()
    performance_monitor = PerformanceMonitor()
    
    # Test frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Performance measurements
    frame_times = []
    gesture_latencies = []
    memory_usage = []
    cpu_usage = []
    
    # Simulate 100 frames of processing
    num_frames = 100
    start_time = time.time()
    
    for frame_num in range(num_frames):
        frame_start = time.time()
        
        # Measure memory and CPU before processing
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        cpu_before = process.cpu_percent()
        
        # Hand detection
        detection_results = hand_detector.detect_hands(test_frame)
        
        # Gesture recognition
        gesture_start = time.time()
        enhanced_results = gesture_recognizer.process_hands(detection_results)
        gesture_end = time.time()
        gesture_latency = (gesture_end - gesture_start) * 1000  # ms
        
        # UI rendering
        performance_monitor.update()
        perf_stats = performance_monitor.get_system_stats()
        
        rendered_frame = overlay_renderer.render_professional_ui(
            test_frame, enhanced_results, perf_stats
        )
        rendered_frame = message_overlay.render_messages(rendered_frame)
        
        frame_end = time.time()
        frame_time = frame_end - frame_start
        
        # Measure memory and CPU after processing
        memory_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        cpu_after = process.cpu_percent()
        
        # Record measurements
        frame_times.append(frame_time)
        gesture_latencies.append(gesture_latency)
        memory_usage.append(memory_after)
        cpu_usage.append(cpu_after)
        
        # Simulate realistic frame rate
        if frame_num < num_frames - 1:
            time.sleep(0.03)  # ~30 FPS simulation
    
    total_time = time.time() - start_time
    
    # Calculate performance metrics
    avg_frame_time = np.mean(frame_times)
    max_frame_time = np.max(frame_times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    avg_gesture_latency = np.mean(gesture_latencies)
    max_gesture_latency = np.max(gesture_latencies)
    
    avg_memory = np.mean(memory_usage)
    max_memory = np.max(memory_usage)
    
    avg_cpu = np.mean(cpu_usage)
    max_cpu = np.max(cpu_usage)
    
    # Display results
    logger.info("📊 Performance Test Results:")
    logger.info(f"🎬 Total frames processed: {num_frames}")
    logger.info(f"⏱️  Total processing time: {total_time:.2f}s")
    logger.info("")
    
    # FPS Performance
    logger.info("🎯 FPS Performance:")
    logger.info(f"   Average FPS: {avg_fps:.1f}")
    logger.info(f"   Target: ≥20 FPS (higher is better)")
    fps_target_met = avg_fps >= 20  # Higher FPS is better, so just check minimum
    logger.info(f"   Status: {'✅ PASSED' if fps_target_met else '❌ FAILED'}")
    logger.info("")
    
    # Gesture Latency
    logger.info("⚡ Gesture Latency:")
    logger.info(f"   Average latency: {avg_gesture_latency:.2f}ms")
    logger.info(f"   Maximum latency: {max_gesture_latency:.2f}ms")
    logger.info(f"   Target: <100ms")
    latency_target_met = avg_gesture_latency < 100 and max_gesture_latency < 100
    logger.info(f"   Status: {'✅ PASSED' if latency_target_met else '❌ FAILED'}")
    logger.info("")
    
    # Memory Usage
    logger.info("🧠 Memory Usage:")
    logger.info(f"   Average memory: {avg_memory:.2f}GB")
    logger.info(f"   Maximum memory: {max_memory:.2f}GB")
    logger.info(f"   Target: <13.5GB")
    memory_target_met = max_memory < 13.5
    logger.info(f"   Status: {'✅ PASSED' if memory_target_met else '❌ FAILED'}")
    logger.info("")
    
    # CPU Usage
    logger.info("🔧 CPU Usage:")
    logger.info(f"   Average CPU: {avg_cpu:.1f}%")
    logger.info(f"   Maximum CPU: {max_cpu:.1f}%")
    logger.info(f"   Target: <40%")
    cpu_target_met = max_cpu < 40
    logger.info(f"   Status: {'✅ PASSED' if cpu_target_met else '❌ FAILED'}")
    logger.info("")
    
    # Overall Assessment
    all_targets_met = fps_target_met and latency_target_met and memory_target_met and cpu_target_met
    
    logger.info("🎯 Overall Performance Assessment:")
    logger.info(f"   FPS Target (≥20): {'✅' if fps_target_met else '❌'}")
    logger.info(f"   Latency Target (<100ms): {'✅' if latency_target_met else '❌'}")
    logger.info(f"   Memory Target (<13.5GB): {'✅' if memory_target_met else '❌'}")
    logger.info(f"   CPU Target (<40%): {'✅' if cpu_target_met else '❌'}")
    logger.info("")
    logger.info(f"🏆 Phase 3 Performance: {'🎉 ALL TARGETS MET!' if all_targets_met else '⚠️  SOME TARGETS NOT MET'}")
    
    # Additional insights
    logger.info("")
    logger.info("💡 Performance Insights:")
    logger.info(f"   Frame processing efficiency: {avg_frame_time*1000:.2f}ms per frame")
    logger.info(f"   Gesture recognition overhead: {avg_gesture_latency:.2f}ms per frame")
    logger.info(f"   Memory efficiency: {avg_memory:.2f}GB baseline usage")
    logger.info(f"   CPU efficiency: {avg_cpu:.1f}% average load")
    
    # Cleanup
    hand_detector.cleanup()
    gesture_recognizer.cleanup()
    
    return {
        'fps': avg_fps,
        'latency': avg_gesture_latency,
        'memory': max_memory,
        'cpu': max_cpu,
        'all_targets_met': all_targets_met
    }


def main():
    """Run performance verification tests."""
    logger.info("🚀 Starting NextSight Phase 3 Performance Verification...")
    
    try:
        results = test_phase3_performance_targets()
        
        if results['all_targets_met']:
            logger.info("🎉 Phase 3 Ready for Production!")
            logger.info("✅ All performance targets successfully met")
            logger.info("🚀 Advanced gesture recognition system validated")
        else:
            logger.warning("⚠️  Performance optimization may be needed")
            logger.info("📊 Review metrics above for optimization opportunities")
        
    except Exception as e:
        logger.error(f"Performance test error: {e}")
        raise


if __name__ == "__main__":
    main()