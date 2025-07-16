"""Performance monitoring utilities."""

import time
import psutil
import threading
from collections import deque
from loguru import logger

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.debug("GPUtil not available - GPU monitoring disabled")


class PerformanceMonitor:
    """Monitor application performance metrics."""
    
    def __init__(self, window_size=30):
        """Initialize performance monitor."""
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        self.start_time = time.time()
        self.total_frames = 0
        self.lock = threading.Lock()
        
        # Performance targets
        self.target_fps = 30
        self.target_latency_ms = 50
        
        # Statistics
        self.min_fps = float('inf')
        self.max_fps = 0
        self.avg_fps = 0
    
    def update(self):
        """Update performance metrics."""
        with self.lock:
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            self.last_frame_time = current_time
            self.total_frames += 1
            
            # Update FPS statistics
            current_fps = self.get_fps()
            if current_fps > 0:
                self.min_fps = min(self.min_fps, current_fps)
                self.max_fps = max(self.max_fps, current_fps)
    
    def get_fps(self):
        """Get current FPS."""
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_avg_fps(self):
        """Get average FPS over entire session."""
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0
    
    def get_frame_latency_ms(self):
        """Get current frame latency in milliseconds."""
        if len(self.frame_times) == 0:
            return 0.0
        return self.frame_times[-1] * 1000  # Convert to ms
    
    def get_gpu_stats(self):
        """Get GPU usage statistics."""
        if not GPU_AVAILABLE:
            return {"available": False}
            
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {"available": False}
                
            gpu = gpus[0]  # Use first GPU (RTX 4050)
            return {
                "available": True,
                "name": gpu.name,
                "load_percent": gpu.load * 100,
                "memory_used_mb": gpu.memoryUsed,
                "memory_total_mb": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "temperature": gpu.temperature
            }
        except Exception as e:
            logger.debug(f"GPU stats error: {e}")
            return {"available": False, "error": str(e)}
    
    def get_system_stats(self):
        """Get comprehensive system resource usage."""
        current_fps = self.get_fps()
        avg_fps = self.get_avg_fps()
        latency_ms = self.get_frame_latency_ms()
        
        # Performance status
        fps_status = "good" if current_fps >= self.target_fps * 0.9 else "degraded"
        latency_status = "good" if latency_ms <= self.target_latency_ms else "high"
        
        stats = {
            # FPS metrics
            'current_fps': current_fps,
            'avg_fps': avg_fps,
            'min_fps': self.min_fps if self.min_fps != float('inf') else 0,
            'max_fps': self.max_fps,
            'fps_status': fps_status,
            
            # Latency metrics
            'latency_ms': latency_ms,
            'latency_status': latency_status,
            'target_fps': self.target_fps,
            'target_latency_ms': self.target_latency_ms,
            
            # System metrics
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            
            # Session metrics
            'uptime_seconds': time.time() - self.start_time,
            'total_frames': self.total_frames,
            
            # GPU metrics
            'gpu': self.get_gpu_stats()
        }
        
        return stats
    
    def log_stats(self, detailed=False):
        """Log performance statistics."""
        stats = self.get_system_stats()
        
        if detailed:
            logger.info(f"Performance Report:")
            logger.info(f"  FPS: {stats['current_fps']:.1f} (avg: {stats['avg_fps']:.1f}, "
                       f"range: {stats['min_fps']:.1f}-{stats['max_fps']:.1f}) [{stats['fps_status']}]")
            logger.info(f"  Latency: {stats['latency_ms']:.1f}ms [{stats['latency_status']}]")
            logger.info(f"  CPU: {stats['cpu_percent']:.1f}%, "
                       f"Memory: {stats['memory_percent']:.1f}% "
                       f"({stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f}GB)")
            
            if stats['gpu']['available']:
                gpu = stats['gpu']
                logger.info(f"  GPU: {gpu['load_percent']:.1f}% load, "
                           f"{gpu['memory_percent']:.1f}% memory "
                           f"({gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB)")
            
            logger.info(f"  Session: {stats['total_frames']} frames in {stats['uptime_seconds']:.1f}s")
        else:
            logger.info(f"Performance - FPS: {stats['current_fps']:.1f} "
                       f"[{stats['fps_status']}], "
                       f"Latency: {stats['latency_ms']:.1f}ms "
                       f"[{stats['latency_status']}], "
                       f"CPU: {stats['cpu_percent']:.1f}%, "
                       f"Memory: {stats['memory_percent']:.1f}%")
    
    def is_performance_good(self):
        """Check if performance meets targets."""
        stats = self.get_system_stats()
        return (stats['fps_status'] == 'good' and 
                stats['latency_status'] == 'good' and
                stats['cpu_percent'] < 80 and
                stats['memory_percent'] < 80)