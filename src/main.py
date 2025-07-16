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
    DISPLAY_FPS, PERFORMANCE_LOG_INTERVAL, DEFAULT_OVERLAY_MODE
)
from core.camera_manager import CameraManager
from detection.hand_detector import HandDetector
from ui.overlay_renderer import OverlayRenderer
from utils.performance_monitor import PerformanceMonitor


class NextSightApp:
    """NextSight Phase 2 - Professional Hand Detection Application."""
    
    def __init__(self):
        """Initialize NextSight Phase 2 application."""
        self.camera_manager = None
        self.hand_detector = None
        self.overlay_renderer = None
        self.performance_monitor = None
        self.running = False
        self.screenshot_counter = 0
        self.fullscreen = False
        
        # Application state
        self.hand_detection_enabled = True
        self.overlay_mode = DEFAULT_OVERLAY_MODE
        
    def initialize(self):
        """Initialize all application components."""
        logger.info("Initializing NextSight Phase 2 components...")
        
        # Initialize camera manager
        self.camera_manager = CameraManager()
        self.camera_manager.start()
        
        # Initialize hand detector
        self.hand_detector = HandDetector()
        
        # Initialize overlay renderer
        self.overlay_renderer = OverlayRenderer()
        self.overlay_renderer.set_overlay_mode(self.overlay_mode)
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Test camera functionality
        test_frame = self.camera_manager.get_frame()
        if test_frame is None:
            raise RuntimeError("Camera test failed - no frame received")
        
        logger.info("All Phase 2 components initialized successfully")
        logger.info("Camera test passed - frame received")
        logger.info("Hand detection system ready")
        
    def run(self):
        """Run the main application loop with hand detection."""
        self.running = True
        last_perf_log = time.time()
        
        logger.info("Starting NextSight Phase 2 main application loop...")
        logger.info("Controls: 'q'=quit, 's'=screenshot, 'h'=toggle hands, " +
                   "'o'=overlay modes, 'f'=fullscreen, 'r'=reset")
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera_manager.get_frame()
                
                if frame is not None:
                    # Update performance metrics
                    self.performance_monitor.update()
                    
                    # Hand detection processing
                    detection_results = self._process_hand_detection(frame)
                    
                    # Get performance stats
                    performance_stats = self.performance_monitor.get_system_stats()
                    
                    # Render professional UI overlay
                    frame = self.overlay_renderer.render_professional_ui(
                        frame, detection_results, performance_stats
                    )
                    
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
        
        if self.overlay_renderer:
            self.overlay_renderer.set_overlay_mode(DEFAULT_OVERLAY_MODE)
            self.overlay_mode = DEFAULT_OVERLAY_MODE
            logger.info("Overlay settings reset to defaults")
    
    def _log_hand_detection_stats(self, detection_results):
        """Log hand detection statistics."""
        if detection_results['hands_detected'] > 0:
            logger.info(f"Hand Detection - Hands: {detection_results['hands_detected']}, "
                       f"Total Fingers: {detection_results['total_fingers']}, "
                       f"Left: {detection_results['left_fingers']}, "
                       f"Right: {detection_results['right_fingers']}, "
                       f"Avg Confidence: {detection_results['confidence_avg']:.2f}")
    
    def _take_screenshot(self, frame):
        """Take a screenshot with full UI and hand overlays."""
        if frame is not None:
            self.screenshot_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"nextsight_phase2_screenshot_{timestamp}_{self.screenshot_counter:03d}.jpg"
            
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
        logger.info("Shutting down NextSight Phase 2 application...")
        
        self.running = False
        
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
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("NextSight Phase 2 application shutdown complete")


def setup_logging():
    """Configure logging for the application."""
    logger.remove()  # Remove default logger
    logger.add(sys.stderr, level=LOG_LEVEL, format=LOG_FORMAT)
    logger.add(LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT, rotation="1 day")


def main():
    """Main application entry point."""
    setup_logging()
    logger.info("Starting NextSight Phase 2 - Professional Hand Detection...")
    
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