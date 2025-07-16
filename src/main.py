"""Main entry point for NextSight Phase 2 - Professional Hand Detection."""

import sys
import cv2
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config.settings import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE, WINDOW_NAME, 
    DISPLAY_FPS, PERFORMANCE_LOG_INTERVAL, DEFAULT_OVERLAY_MODE, UI_THEME
)
from core.camera_manager import CameraManager
from detection.hand_detector import HandDetector
from detection.gesture_recognizer import GestureRecognizer
from ui.overlay_renderer import OverlayRenderer
from ui.message_overlay import MessageOverlay
from utils.performance_monitor import PerformanceMonitor


class NextSightApp:
    """NextSight Phase 3 - Advanced Gesture Recognition Application."""
    
    def __init__(self):
        """Initialize NextSight Phase 3 application."""
        self.camera_manager = None
        self.hand_detector = None
        self.gesture_recognizer = None
        self.overlay_renderer = None
        self.message_overlay = None
        self.performance_monitor = None
        self.running = False
        self.screenshot_counter = 0
        self.fullscreen = False
        
        # Application state
        self.hand_detection_enabled = True
        self.gesture_recognition_enabled = True
        self.message_overlay_enabled = True
        self.overlay_mode = DEFAULT_OVERLAY_MODE
        self.sensitivity_level = 'medium'  # low, medium, high
        
    def initialize(self):
        """Initialize all application components."""
        logger.info("Initializing NextSight Phase 3 components...")
        
        # Initialize camera manager
        self.camera_manager = CameraManager()
        self.camera_manager.start()
        
        # Initialize hand detector
        self.hand_detector = HandDetector()
        
        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer()
        
        # Initialize overlay renderer
        self.overlay_renderer = OverlayRenderer()
        self.overlay_renderer.set_overlay_mode(self.overlay_mode)
        
        # Initialize message overlay
        self.message_overlay = MessageOverlay()
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Test camera functionality
        test_frame = self.camera_manager.get_frame()
        if test_frame is None:
            raise RuntimeError("Camera test failed - no frame received")
        
        logger.info("All Phase 3 components initialized successfully")
        logger.info("Camera test passed - frame received")
        logger.info("Hand detection system ready")
        logger.info("Gesture recognition system ready")
        logger.info("Interactive message system ready")
        
    def run(self):
        """Run the main application loop with hand detection."""
        self.running = True
        last_perf_log = time.time()
        
        logger.info("Starting NextSight Phase 3 main application loop...")
        logger.info("Controls: 'q'=quit, 's'=screenshot, 'h'=toggle hands, " +
                   "'o'=overlay modes, 'f'=fullscreen, 'r'=reset")
        logger.info("Gesture Controls: 'g'=toggle gestures, 'm'=toggle messages, " +
                   "'c'=clear history, 't'=adjust sensitivity")
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera_manager.get_frame()
                
                if frame is not None:
                    # Update performance metrics
                    self.performance_monitor.update()
                    
                    # Hand detection processing
                    detection_results = self._process_hand_detection(frame)
                    
                    # Gesture recognition processing
                    detection_results = self._process_gesture_recognition(detection_results)
                    
                    # Get performance stats
                    performance_stats = self.performance_monitor.get_system_stats()
                    
                    # Render professional UI overlay
                    frame = self.overlay_renderer.render_professional_ui(
                        frame, detection_results, performance_stats
                    )
                    
                    # Render gesture messages
                    frame = self._render_gesture_messages(frame, detection_results)
                    
                    # Display frame
                    if self.fullscreen:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    
                    cv2.imshow(WINDOW_NAME, frame)
                    
                    # Check camera health
                    if not self.camera_manager.is_camera_healthy():
                        logger.warning("Camera health check failed")
                
                # Handle keyboard input with enhanced controls
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard_input(key, frame):
                    break
                
                # Log performance statistics periodically
                if time.time() - last_perf_log > PERFORMANCE_LOG_INTERVAL:
                    self.performance_monitor.log_stats()
                    self._log_hand_detection_stats(detection_results)
                    self._log_gesture_stats(detection_results)
                    last_perf_log = time.time()
                    
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error in main loop: {e}")
            raise
        finally:
            self.shutdown()
    
    def _process_hand_detection(self, frame):
        """Process hand detection on the current frame."""
        if self.hand_detection_enabled and self.hand_detector.is_active():
            return self.hand_detector.detect_hands(frame)
        else:
            return {
                'hands_detected': 0,
                'hands': [],
                'total_fingers': 0,
                'left_fingers': 0,
                'right_fingers': 0,
                'raw_results': None,
                'confidence_avg': 0.0
            }
    
    def _process_gesture_recognition(self, detection_results):
        """Process gesture recognition on hand detection results."""
        if self.gesture_recognition_enabled and self.gesture_recognizer.is_enabled():
            return self.gesture_recognizer.process_hands(detection_results)
        else:
            # Add empty gesture results
            detection_results['gesture_recognition'] = {
                'enabled': False,
                'detection_paused': False,
                'raw_detections': {},
                'confirmed_events': [],
                'current_gestures': {'Left': None, 'Right': None},
                'session_stats': {'total_gestures': 0, 'average_confidence': 0.0},
                'cooldown_status': {'Left': {}, 'Right': {}}
            }
            return detection_results
    
    def _render_gesture_messages(self, frame, detection_results):
        """Render gesture messages and status overlays."""
        if not self.message_overlay_enabled:
            return frame
        
        # Add gesture messages for confirmed events
        gesture_info = detection_results.get('gesture_recognition', {})
        confirmed_events = gesture_info.get('confirmed_events', [])
        
        for event in confirmed_events:
            self.message_overlay.add_gesture_message(
                event.gesture_type, event.hand_label, event.confidence
            )
        
        # Render messages
        frame = self.message_overlay.render_messages(frame)
        
        # Render gesture status overlay
        frame = self.message_overlay.render_gesture_status(frame, gesture_info)
        
        return frame
    
    def _handle_keyboard_input(self, key, frame) -> bool:
        """Handle enhanced keyboard input. Returns False to quit."""
        if key == ord('q') or key == 27:  # 'q' or ESC
            logger.info("Quit requested by user")
            return False
        elif key == ord('s'):
            self._take_screenshot(frame)
        elif key == ord('h'):
            self._toggle_hand_detection()
        elif key == ord('o'):
            self._cycle_overlay_mode()
        elif key == ord('f'):
            self._toggle_fullscreen()
        elif key == ord('r'):
            self._reset_detection_settings()
        # Phase 3 Gesture Controls
        elif key == ord('g'):
            self._toggle_gesture_recognition()
        elif key == ord('m'):
            self._toggle_message_overlay()
        elif key == ord('c'):
            self._clear_gesture_history()
        elif key == ord('t'):
            self._cycle_sensitivity()
        
        return True
    
    def _toggle_hand_detection(self):
        """Toggle hand detection on/off."""
        if self.hand_detector:
            self.hand_detection_enabled = self.hand_detector.toggle_detection()
            status = "enabled" if self.hand_detection_enabled else "disabled"
            logger.info(f"Hand detection {status}")
    
    def _cycle_overlay_mode(self):
        """Cycle through overlay display modes."""
        if self.overlay_renderer:
            new_mode = self.overlay_renderer.cycle_overlay_mode()
            self.overlay_mode = new_mode
            logger.info(f"Overlay mode: {new_mode}")
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        status = "enabled" if self.fullscreen else "disabled"
        logger.info(f"Fullscreen mode {status}")
    
    def _reset_detection_settings(self):
        """Reset detection settings to defaults."""
        if self.hand_detector:
            self.hand_detection_enabled = True
            # Reset hand detector state
            logger.info("Detection settings reset to defaults")
        
        if self.gesture_recognizer:
            self.gesture_recognition_enabled = True
            self.gesture_recognizer.clear_gesture_history()
            logger.info("Gesture recognition reset to defaults")
        
        if self.overlay_renderer:
            self.overlay_renderer.set_overlay_mode(DEFAULT_OVERLAY_MODE)
            self.overlay_mode = DEFAULT_OVERLAY_MODE
            logger.info("Overlay settings reset to defaults")
        
        if self.message_overlay:
            self.message_overlay.clear_messages()
            logger.info("Message overlay cleared")
    
    def _toggle_gesture_recognition(self):
        """Toggle gesture recognition on/off."""
        if self.gesture_recognizer:
            self.gesture_recognition_enabled = self.gesture_recognizer.toggle_recognition()
            status = "enabled" if self.gesture_recognition_enabled else "disabled"
            logger.info(f"Gesture recognition {status}")
            
            # Add visual feedback
            if self.message_overlay:
                self.message_overlay.add_custom_message(
                    f"Gesture Recognition {status.title()}", 
                    UI_THEME['accent'] if self.gesture_recognition_enabled else UI_THEME['error']
                )
    
    def _toggle_message_overlay(self):
        """Toggle message overlay on/off."""
        if self.message_overlay:
            self.message_overlay_enabled = self.message_overlay.toggle_messages()
            status = "enabled" if self.message_overlay_enabled else "disabled"
            logger.info(f"Message overlay {status}")
    
    def _clear_gesture_history(self):
        """Clear gesture history and reset counters."""
        if self.gesture_recognizer:
            self.gesture_recognizer.clear_gesture_history()
            logger.info("Gesture history cleared")
            
            # Add visual feedback
            if self.message_overlay:
                self.message_overlay.add_custom_message(
                    "Gesture History Cleared", UI_THEME['warning']
                )
    
    def _cycle_sensitivity(self):
        """Cycle through gesture sensitivity levels."""
        sensitivity_levels = ['low', 'medium', 'high']
        current_index = sensitivity_levels.index(self.sensitivity_level)
        next_index = (current_index + 1) % len(sensitivity_levels)
        self.sensitivity_level = sensitivity_levels[next_index]
        
        if self.gesture_recognizer:
            self.gesture_recognizer.adjust_sensitivity(self.sensitivity_level)
            logger.info(f"Gesture sensitivity set to {self.sensitivity_level}")
            
            # Add visual feedback
            if self.message_overlay:
                self.message_overlay.add_custom_message(
                    f"Sensitivity: {self.sensitivity_level.title()}", UI_THEME['accent']
                )
    
    def _log_hand_detection_stats(self, detection_results):
        """Log hand detection statistics."""
        if detection_results['hands_detected'] > 0:
            logger.info(f"Hand Detection - Hands: {detection_results['hands_detected']}, "
                       f"Total Fingers: {detection_results['total_fingers']}, "
                       f"Left: {detection_results['left_fingers']}, "
                       f"Right: {detection_results['right_fingers']}, "
                       f"Avg Confidence: {detection_results['confidence_avg']:.2f}")
    
    def _log_gesture_stats(self, detection_results):
        """Log gesture recognition statistics."""
        gesture_info = detection_results.get('gesture_recognition', {})
        if not gesture_info.get('enabled', False):
            return
        
        # Log confirmed events
        confirmed_events = gesture_info.get('confirmed_events', [])
        for event in confirmed_events:
            logger.info(f"Gesture Event - {event.gesture_type} ({event.hand_label}) - "
                       f"Confidence: {event.confidence:.2f}, Duration: {event.duration:.2f}s")
        
        # Log session stats periodically
        stats = gesture_info.get('session_stats', {})
        if stats.get('total_gestures', 0) > 0:
            logger.info(f"Gesture Session - Total: {stats['total_gestures']}, "
                       f"Avg Confidence: {stats['average_confidence']:.2f}, "
                       f"Rate: {stats.get('gestures_per_minute', 0):.1f}/min")
    
    def _take_screenshot(self, frame):
        """Take a screenshot with full UI and hand overlays."""
        if frame is not None:
            self.screenshot_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"nextsight_phase3_screenshot_{timestamp}_{self.screenshot_counter:03d}.jpg"
            
            try:
                cv2.imwrite(filename, frame)
                logger.info(f"Professional screenshot saved: {filename}")
                
                # Show confirmation briefly
                temp_frame = frame.copy()
                cv2.putText(temp_frame, f"Screenshot saved: {filename}", 
                           (10, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, temp_frame)
                cv2.waitKey(1000)  # Show for 1 second
                
            except Exception as e:
                logger.error(f"Failed to save screenshot: {e}")
    
    def shutdown(self):
        """Shutdown application and cleanup resources."""
        logger.info("Shutting down NextSight Phase 3 application...")
        
        self.running = False
        
        # Cleanup gesture recognizer
        if self.gesture_recognizer:
            self.gesture_recognizer.cleanup()
        
        # Cleanup hand detector
        if self.hand_detector:
            self.hand_detector.cleanup()
        
        # Cleanup camera
        if self.camera_manager:
            self.camera_manager.stop()
        
        # Log final performance statistics
        if self.performance_monitor:
            logger.info("Final performance report:")
            self.performance_monitor.log_stats(detailed=True)
        
        # Log final gesture statistics
        if self.gesture_recognizer:
            stats = self.gesture_recognizer.get_session_stats()
            logger.info(f"Final gesture statistics - Total: {stats['total_gestures']}, "
                       f"Avg Confidence: {stats['average_confidence']:.2f}")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("NextSight Phase 3 application shutdown complete")


def setup_logging():
    """Configure logging for the application."""
    logger.remove()  # Remove default logger
    logger.add(sys.stderr, level=LOG_LEVEL, format=LOG_FORMAT)
    logger.add(LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT, rotation="1 day")


def main():
    """Main application entry point."""
    setup_logging()
    logger.info("Starting NextSight Phase 3 - Advanced Gesture Recognition...")
    
    app = NextSightApp()
    
    try:
        app.initialize()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()