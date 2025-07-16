"""Camera management for NextSight."""

import cv2
import time
import numpy as np
from loguru import logger
from config.settings import (
    CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, 
    CAMERA_BUFFER_SIZE, CAMERA_FORMAT, CAMERA_AUTO_EXPOSURE,
    MAX_FRAME_FAILURES, CAMERA_RECONNECT_ATTEMPTS, CAMERA_RECONNECT_DELAY,
    MIRROR_DISPLAY
)


class CameraManager:
    """Manages camera capture and video stream."""
    
    def __init__(self, camera_id=CAMERA_ID, mock_mode=False):
        """Initialize camera manager."""
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.frame_failures = 0
        self.last_frame = None
        self.camera_info = {}
        self.mock_mode = mock_mode
        self.mock_frame_counter = 0
        
    def start(self):
        """Start camera capture with optimal settings for RTX 4050."""
        try:
            logger.info(f"Initializing camera {self.camera_id}...")
            
            if self.mock_mode:
                logger.info("Starting in mock mode (no camera available)")
                self._setup_mock_camera()
            else:
                # Try to initialize real camera
                try:
                    self._setup_real_camera()
                except Exception as e:
                    logger.warning(f"Real camera failed: {e}, falling back to mock mode")
                    self.mock_mode = True
                    self._setup_mock_camera()
            
            # Get camera info for diagnostics
            self._get_camera_info()
            
            self.is_running = True
            self.frame_failures = 0
            
            logger.info(f"Camera {self.camera_id} initialized successfully")
            logger.info(f"Camera info: {self.camera_info}")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            self.stop()
            raise
    
    def _setup_real_camera(self):
        """Setup real camera hardware."""
        # Initialize camera with DirectShow backend for better Windows performance
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Configure camera settings for optimal performance
        self._configure_camera()
    
    def _setup_mock_camera(self):
        """Setup mock camera for testing."""
        self.cap = None  # No actual capture device
        logger.info("Mock camera setup complete")
    
    def _configure_camera(self):
        """Configure camera with optimal settings."""
        if self.cap is None:
            return
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Set buffer size for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        
        # Set format to MJPG for better compression
        fourcc = cv2.VideoWriter_fourcc(*CAMERA_FORMAT)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Enable auto exposure for varying lighting
        if CAMERA_AUTO_EXPOSURE:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        
        # Additional performance optimizations
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Keep in BGR for OpenCV
        
        logger.debug("Camera configuration applied")
    
    def _get_camera_info(self):
        """Get camera information for diagnostics."""
        if self.mock_mode:
            self.camera_info = {
                "mode": "mock",
                "width": CAMERA_WIDTH,
                "height": CAMERA_HEIGHT,
                "fps": CAMERA_FPS,
                "buffer_size": CAMERA_BUFFER_SIZE,
                "format": "MOCK",
                "auto_exposure": "N/A",
                "backend": "Mock Backend"
            }
        elif self.cap is not None:
            self.camera_info = {
                "mode": "real",
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "buffer_size": int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE)),
                "format": self.cap.get(cv2.CAP_PROP_FOURCC),
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "backend": self.cap.getBackendName()
            }
    
    def _generate_mock_frame(self):
        """Generate a mock frame for testing."""
        # Create a colorful test pattern
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(CAMERA_HEIGHT):
            frame[i, :] = [
                int(255 * i / CAMERA_HEIGHT),  # Red gradient
                int(255 * (1 - i / CAMERA_HEIGHT)),  # Green gradient 
                128  # Constant blue
            ]
        
        # Add moving circle
        center_x = int(CAMERA_WIDTH // 2 + 200 * np.sin(self.mock_frame_counter * 0.1))
        center_y = int(CAMERA_HEIGHT // 2 + 100 * np.cos(self.mock_frame_counter * 0.1))
        cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)
        
        # Add frame counter text
        cv2.putText(frame, f"MOCK CAMERA - Frame {self.mock_frame_counter}", 
                   (10, CAMERA_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (CAMERA_WIDTH - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.mock_frame_counter += 1
        return frame
    
    def get_frame(self):
        """Get current frame from camera with error handling."""
        if not self.is_running:
            logger.warning("Camera not running")
            return None
        
        try:
            if self.mock_mode:
                # Generate mock frame
                frame = self._generate_mock_frame()
                time.sleep(1.0 / CAMERA_FPS)  # Simulate camera timing
            else:
                # Get real frame
                if self.cap is None:
                    logger.warning("Camera not initialized")
                    return None
                    
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.frame_failures += 1
                    logger.warning(f"Frame read failed (attempt {self.frame_failures}/{MAX_FRAME_FAILURES})")
                    
                    if self.frame_failures >= MAX_FRAME_FAILURES:
                        logger.error("Too many frame failures, attempting camera restart")
                        self._restart_camera()
                        
                    return self.last_frame  # Return last good frame
            
            # Reset failure counter on successful read
            self.frame_failures = 0
            
            # Apply mirror effect for better demo experience
            if MIRROR_DISPLAY:
                frame = cv2.flip(frame, 1)
            
            # Store as last good frame
            self.last_frame = frame.copy()
            
            return frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            self.frame_failures += 1
            return self.last_frame
    
    def _restart_camera(self):
        """Restart camera after failures."""
        logger.info("Restarting camera...")
        self.stop()
        
        for attempt in range(CAMERA_RECONNECT_ATTEMPTS):
            try:
                time.sleep(CAMERA_RECONNECT_DELAY)
                self.start()
                logger.info("Camera restarted successfully")
                return
            except Exception as e:
                logger.warning(f"Camera restart attempt {attempt + 1} failed: {e}")
        
        logger.error("Failed to restart camera after all attempts")
        self.is_running = False
    
    def stop(self):
        """Stop camera capture and cleanup."""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        logger.info("Camera stopped and resources released")
    
    def get_frame_for_detection(self):
        """Get frame optimized for hand detection processing."""
        frame = self.get_frame()
        if frame is None:
            return None
        
        # Apply preprocessing for optimal hand detection
        # MediaPipe works best with RGB, but we'll convert in the detector
        # Here we can add any camera-specific preprocessing
        
        return frame
    
    def get_camera_info(self):
        """Get camera information."""
        return self.camera_info.copy()
    
    def is_camera_healthy(self):
        """Check if camera is healthy."""
        if self.mock_mode:
            return self.is_running
        else:
            return (self.is_running and 
                    self.cap is not None and 
                    self.cap.isOpened() and 
                    self.frame_failures < MAX_FRAME_FAILURES)