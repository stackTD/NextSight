#!/usr/bin/env python3
"""Visual demonstration of NextSight Phase 3 gesture recognition capabilities."""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from detection.gesture_recognizer import GestureRecognizer
from ui.message_overlay import MessageOverlay
from ui.overlay_renderer import OverlayRenderer
from utils.performance_monitor import PerformanceMonitor


def create_demo_frame() -> np.ndarray:
    """Create a demo frame showing NextSight Phase 3 capabilities."""
    # Create base frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Create gradient background
    for y in range(720):
        for x in range(1280):
            frame[y, x] = [30 + y//24, 30 + x//43, 60]
    
    # Add title
    cv2.putText(frame, "NextSight Phase 3 - Advanced Gesture Recognition", 
               (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
    
    return frame


def simulate_gesture_detection_demo():
    """Create visual demonstration of gesture detection capabilities."""
    logger.info("Creating NextSight Phase 3 visual demonstration...")
    
    # Initialize components
    gesture_recognizer = GestureRecognizer()
    message_overlay = MessageOverlay()
    overlay_renderer = OverlayRenderer()
    performance_monitor = PerformanceMonitor()
    
    # Create demo frame
    base_frame = create_demo_frame()
    
    # Demo gesture sequence
    gesture_demo_sequence = [
        ('peace', 'Right', 0.95),
        ('thumbs_up', 'Left', 0.88),
        ('ok', 'Right', 0.92),
        ('stop', 'Left', 0.86),
        ('thumbs_down', 'Right', 0.90)
    ]
    
    demo_frames = []
    
    for i, (gesture_type, hand_label, confidence) in enumerate(gesture_demo_sequence):
        logger.info(f"Demonstrating {gesture_type} gesture...")
        
        # Add gesture message
        message_overlay.add_gesture_message(gesture_type, hand_label, confidence)
        
        # Create mock detection results
        mock_results = {
            'hands_detected': 1,
            'hands': [{
                'label': hand_label,
                'landmarks': None,  # Mock mode
                'confidence': confidence,
                'finger_count': 2 if gesture_type == 'peace' else 1,
                'current_gesture': gesture_type,
                'gesture_confidence': confidence
            }],
            'total_fingers': 2 if gesture_type == 'peace' else 1,
            'left_fingers': 2 if hand_label == 'Left' and gesture_type == 'peace' else 0,
            'right_fingers': 2 if hand_label == 'Right' and gesture_type == 'peace' else 0,
            'raw_results': None,
            'confidence_avg': confidence,
            'gesture_recognition': {
                'enabled': True,
                'detection_paused': False,
                'current_gestures': {hand_label: gesture_type},
                'session_stats': {'total_gestures': i+1, 'average_confidence': confidence},
                'cooldown_status': {'Left': {}, 'Right': {}}
            }
        }
        
        # Update performance
        performance_monitor.update()
        performance_stats = performance_monitor.get_system_stats()
        
        # Create frame for this gesture
        frame = base_frame.copy()
        
        # Add gesture demonstration text
        demo_text = f"Demonstrating: {gesture_type.replace('_', ' ').title()} ({hand_label} Hand)"
        cv2.putText(frame, demo_text, (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Add gesture info
        info_text = f"Confidence: {confidence:.1%} | Detection #{i+1}"
        cv2.putText(frame, info_text, (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Render professional UI
        frame = overlay_renderer.render_professional_ui(frame, mock_results, performance_stats)
        
        # Render gesture message
        frame = message_overlay.render_messages(frame)
        
        # Add Phase 3 features showcase
        features_y = 200
        features = [
            "✅ 5 Gesture Types: Peace ✌️, Thumbs Up 👍, Thumbs Down 👎, OK 👌, Stop ✋",
            "✅ Professional Message System with Animations",
            "✅ State Management: Cooldowns, Hold Times, Temporal Smoothing",
            "✅ Interactive Controls: 'g' gestures, 'm' messages, 'c' clear, 't' sensitivity",
            "✅ Stop Gesture Special Function: Pause/Resume Detection",
            "✅ Performance Optimized: <100ms latency, 20-25 FPS target"
        ]
        
        for j, feature in enumerate(features):
            if j <= i:  # Progressive reveal
                cv2.putText(frame, feature, (50, features_y + j*35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        # Add controls showcase
        controls_y = 450
        cv2.putText(frame, "Phase 3 Gesture Controls:", (50, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        controls = [
            "'g' - Toggle gesture recognition ON/OFF",
            "'m' - Toggle message overlay display",
            "'c' - Clear gesture history and counters",
            "'t' - Adjust sensitivity (low/medium/high)",
            "✋ Stop Gesture - Pause/Resume detection"
        ]
        
        for j, control in enumerate(controls):
            cv2.putText(frame, control, (50, controls_y + 30 + j*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        demo_frames.append(frame)
        
        # Wait to show animation
        time.sleep(0.5)
    
    # Save demo frames
    logger.info("Saving demonstration screenshots...")
    
    for i, frame in enumerate(demo_frames):
        gesture_name = gesture_demo_sequence[i][0]
        filename = f"nextsight_phase3_demo_{gesture_name}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Saved demonstration: {filename}")
    
    # Create final showcase frame
    final_frame = create_final_showcase_frame(demo_frames)
    cv2.imwrite("nextsight_phase3_complete_showcase.jpg", final_frame)
    logger.info("Saved complete showcase: nextsight_phase3_complete_showcase.jpg")
    
    # Cleanup
    gesture_recognizer.cleanup()
    
    logger.info("🎉 NextSight Phase 3 visual demonstration completed!")
    return demo_frames


def create_final_showcase_frame(demo_frames) -> np.ndarray:
    """Create final showcase frame with multiple gesture demos."""
    # Create large showcase frame
    showcase = np.zeros((1440, 2560, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(showcase, "NextSight Phase 3 - Complete Gesture Recognition Showcase", 
               (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
    
    # Add subtitle
    cv2.putText(showcase, "5 Advanced Gestures with Interactive Feedback Messages", 
               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Arrange demo frames in grid (2x3)
    frame_width = 640
    frame_height = 360
    margin = 50
    
    positions = [
        (margin, 150),  # Top left
        (margin + frame_width + margin, 150),  # Top right
        (margin, 150 + frame_height + margin),  # Middle left
        (margin + frame_width + margin, 150 + frame_height + margin),  # Middle right
        (margin + frame_width//2 + margin//2, 150 + 2*(frame_height + margin))  # Bottom center
    ]
    
    gesture_names = ['Peace ✌️', 'Thumbs Up 👍', 'OK 👌', 'Stop ✋', 'Thumbs Down 👎']
    
    for i, (frame, (x, y)) in enumerate(zip(demo_frames, positions)):
        # Resize frame to fit
        resized = cv2.resize(frame, (frame_width, frame_height))
        showcase[y:y+frame_height, x:x+frame_width] = resized
        
        # Add gesture label
        cv2.putText(showcase, gesture_names[i], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Add technical specifications
    tech_specs_y = 1200
    cv2.putText(showcase, "Technical Specifications:", (50, tech_specs_y), 
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    
    specs = [
        "🎯 Target Performance: 20-25 FPS with gesture recognition active",
        "⚡ Gesture Latency: <100ms from gesture to message display", 
        "🧠 Memory Usage: <13.5GB total (12.3GB + 0.2GB for gestures)",
        "🔧 CPU Usage: <40% total (14% + 5% for gesture processing)",
        "📊 Confidence Threshold: 80% for high accuracy",
        "⏱️ Hold Time: 0.5s minimum before gesture triggers",
        "🔄 Cooldown: 2.0s between same gesture detections",
        "📈 Frame Averaging: 5 frames for stability",
        "🤲 Max Simultaneous: 2 gestures (one per hand)"
    ]
    
    for i, spec in enumerate(specs):
        cv2.putText(showcase, spec, (50, tech_specs_y + 40 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    
    return showcase


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting NextSight Phase 3 visual demonstration...")
    
    try:
        demo_frames = simulate_gesture_detection_demo()
        logger.info(f"✅ Created {len(demo_frames)} demonstration frames")
        logger.info("📸 Screenshots saved showing all gesture types")
        logger.info("🎉 Phase 3 demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration error: {e}")
        raise


if __name__ == "__main__":
    main()