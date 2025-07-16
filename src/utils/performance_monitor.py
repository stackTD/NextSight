"""Performance monitoring utilities."""

import time
import psutil
from collections import deque
from loguru import logger


class PerformanceMonitor:
    """Monitor application performance metrics."""
    
    def __init__(self, window_size=30):
        """Initialize performance monitor."""
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        self.start_time = time.time()
    
    def update(self):
        """Update performance metrics."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
    
    def get_fps(self):
        """Get current FPS."""
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_system_stats(self):
        """Get system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'fps': self.get_fps(),
            'uptime': time.time() - self.start_time
        }
    
    def log_stats(self):
        """Log performance statistics."""
        stats = self.get_system_stats()
        logger.info(f"Performance - FPS: {stats['fps']:.1f}, "
                   f"CPU: {stats['cpu_percent']:.1f}%, "
                   f"Memory: {stats['memory_percent']:.1f}%")