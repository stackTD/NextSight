"""Main entry point for NextSight application."""

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
    DISPLAY_FPS, PERFORMANCE_LOG_INTERVAL
)
from core.camera_manager import CameraManager
from utils.performance_monitor import PerformanceMonitor


class NextSightApp:
    """Main NextSight application class."""
    
    def __init__(self):
        """Initialize NextSight application."""
        self.camera_manager = None
        self.performance_monitor = None
        self.running = False
        self.screenshot_counter = 0
        
    def initialize(self):
        """Initialize all application components."""
        logger.info("Initializing NextSight components...")
        
        # Initialize camera manager
        self.camera_manager = CameraManager()
        self.camera_manager.start()
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Test camera functionality
        test_frame = self.camera_manager.get_frame()
        if test_frame is None:
            raise RuntimeError("Camera test failed - no frame received")
        
        logger.info("All components initialized successfully")
        logger.info("Camera test passed - frame received")
        
    def run(self):
        """Run the main application loop."""
        self.running = True
        last_perf_log = time.time()
        
        logger.info("Starting main application loop...")
        logger.info("Controls: 'q' to quit, 's' for screenshot")
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera_manager.get_frame()
                
                if frame is not None:
                    # Update performance metrics
                    self.performance_monitor.update()
                    
                    # Add FPS overlay if enabled
                    if DISPLAY_FPS:
                        self._add_fps_overlay(frame)
                    
                    # Display frame
                    cv2.imshow(WINDOW_NAME, frame)
                    
                    # Check camera health
                    if not self.camera_manager.is_camera_healthy():
                        logger.warning("Camera health check failed")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    self._take_screenshot(frame)
                
                # Log performance statistics periodically
                if time.time() - last_perf_log > PERFORMANCE_LOG_INTERVAL:
                    self.performance_monitor.log_stats()
                    last_perf_log = time.time()
                    
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error in main loop: {e}")
            raise
        finally:
            self.shutdown()
    
    def _add_fps_overlay(self, frame):
        """Add FPS overlay to frame."""
        fps = self.performance_monitor.get_fps()
        latency = self.performance_monitor.get_frame_latency_ms()
        
        # Choose color based on performance
        if fps >= 25:
            color = (0, 255, 0)  # Green
        elif fps >= 15:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Add FPS text
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
        
        # Add latency text
        latency_text = f"Latency: {latency:.1f}ms"
        cv2.putText(frame, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
        
        # Add camera health indicator
        health_text = "CAM: OK" if self.camera_manager.is_camera_healthy() else "CAM: ERROR"
        health_color = (0, 255, 0) if self.camera_manager.is_camera_healthy() else (0, 0, 255)
        cv2.putText(frame, health_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, health_color, 2)
    
    def _take_screenshot(self, frame):
        """Take a screenshot of the current frame."""
        if frame is not None:
            self.screenshot_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{self.screenshot_counter:03d}.jpg"
            
            try:
                cv2.imwrite(filename, frame)
                logger.info(f"Screenshot saved: {filename}")
                
                # Show confirmation on frame briefly
                cv2.putText(frame, f"Screenshot saved: {filename}", (10, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(500)  # Show for 500ms
                
            except Exception as e:
                logger.error(f"Failed to save screenshot: {e}")
    
    def shutdown(self):
        """Shutdown application and cleanup resources."""
        logger.info("Shutting down NextSight application...")
        
        self.running = False
        
        # Cleanup camera
        if self.camera_manager:
            self.camera_manager.stop()
        
        # Log final performance statistics
        if self.performance_monitor:
            logger.info("Final performance report:")
            self.performance_monitor.log_stats(detailed=True)
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("NextSight application shutdown complete")


def setup_logging():
    """Configure logging for the application."""
    logger.remove()  # Remove default logger
    logger.add(sys.stderr, level=LOG_LEVEL, format=LOG_FORMAT)
    logger.add(LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT, rotation="1 day")


def main():
    """Main application entry point."""
    setup_logging()
    logger.info("Starting NextSight application...")
    
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