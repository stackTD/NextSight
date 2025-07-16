"""Display management for NextSight."""

import cv2
from loguru import logger
from config.settings import WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_FPS


class DisplayManager:
    """Manages the main display window and rendering."""
    
    def __init__(self):
        """Initialize display manager."""
        self.window_name = "NextSight"
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.show_fps = DISPLAY_FPS
        self.is_running = False
        logger.info("Display manager initialized")
    
    def create_window(self):
        """Create the main display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        logger.info(f"Created display window: {self.window_name}")
    
    def show_frame(self, frame):
        """Display a frame in the window."""
        if frame is not None:
            cv2.imshow(self.window_name, frame)
    
    def check_exit(self):
        """Check if user wants to exit."""
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q') or key == 27  # 'q' or ESC
    
    def cleanup(self):
        """Clean up display resources."""
        cv2.destroyAllWindows()
        logger.info("Display manager cleaned up")
    
    def resize_window(self, width, height):
        """Resize the display window."""
        self.width = width
        self.height = height
        cv2.resizeWindow(self.window_name, width, height)