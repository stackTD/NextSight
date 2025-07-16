"""Gesture state management for NextSight Phase 3."""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from config.settings import (
    GESTURE_CONFIDENCE_THRESHOLD, GESTURE_HOLD_TIME, 
    GESTURE_COOLDOWN_TIME, GESTURE_FRAME_AVERAGING
)


@dataclass
class GestureEvent:
    """Represents a detected gesture event."""
    gesture_type: str
    hand_label: str
    confidence: float
    timestamp: float
    duration: float = 0.0


@dataclass
class GestureHistory:
    """Tracks gesture detection history."""
    gesture_events: List[GestureEvent] = field(default_factory=list)
    total_detections: Dict[str, int] = field(default_factory=dict)
    detection_accuracy: Dict[str, float] = field(default_factory=dict)


class GestureState:
    """Intelligent gesture state tracking system."""
    
    def __init__(self):
        """Initialize gesture state management."""
        self.confidence_threshold = GESTURE_CONFIDENCE_THRESHOLD
        self.hold_time = GESTURE_HOLD_TIME
        self.cooldown_time = GESTURE_COOLDOWN_TIME
        self.frame_averaging = GESTURE_FRAME_AVERAGING
        
        # Current gesture states per hand
        self.current_gestures: Dict[str, Optional[str]] = {
            'Left': None,
            'Right': None
        }
        
        # Gesture detection buffers for stability
        self.gesture_buffers: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
            'Left': {},
            'Right': {}
        }
        
        # Cooldown tracking
        self.last_detection_times: Dict[str, Dict[str, float]] = {
            'Left': {},
            'Right': {}
        }
        
        # Hold time tracking
        self.gesture_start_times: Dict[str, Dict[str, float]] = {
            'Left': {},
            'Right': {}
        }
        
        # Session statistics
        self.history = GestureHistory()
        self.session_stats = {
            'total_gestures': 0,
            'session_start': time.time(),
            'gestures_per_minute': 0.0,
            'average_confidence': 0.0
        }
        
        # Detection pause state
        self.detection_paused = False
        self.pause_start_time = None
        
        logger.info("Gesture state management initialized")
        logger.info(f"Thresholds - Confidence: {self.confidence_threshold}, "
                   f"Hold: {self.hold_time}s, Cooldown: {self.cooldown_time}s")
    
    def update_gesture_detection(self, gesture_results: Dict[str, Dict[str, Tuple[bool, float]]]) -> List[GestureEvent]:
        """
        Update gesture state with new detection results.
        
        Args:
            gesture_results: Dict with hand labels as keys, gesture detection results as values
                           Format: {'Left': {'peace': (True, 0.9), 'thumbs_up': (False, 0.3), ...}}
        
        Returns:
            List of confirmed gesture events
        """
        if self.detection_paused:
            return []
        
        confirmed_events = []
        current_time = time.time()
        
        for hand_label, gestures in gesture_results.items():
            for gesture_type, (detected, confidence) in gestures.items():
                # Process each gesture detection
                event = self._process_gesture(hand_label, gesture_type, detected, confidence, current_time)
                if event:
                    confirmed_events.append(event)
        
        # Update session statistics
        self._update_session_stats(confirmed_events)
        
        return confirmed_events
    
    def _process_gesture(self, hand_label: str, gesture_type: str, detected: bool, 
                        confidence: float, current_time: float) -> Optional[GestureEvent]:
        """Process individual gesture detection with stability and cooldown checks."""
        
        # Initialize buffers if needed
        if gesture_type not in self.gesture_buffers[hand_label]:
            self.gesture_buffers[hand_label][gesture_type] = []
        
        buffer = self.gesture_buffers[hand_label][gesture_type]
        
        # Add current detection to buffer
        buffer.append((detected, confidence))
        
        # Maintain buffer size
        if len(buffer) > self.frame_averaging:
            buffer.pop(0)
        
        # Check if we have enough frames for stability
        if len(buffer) < self.frame_averaging:
            return None
        
        # Calculate average confidence and detection rate
        detections = [d for d, c in buffer if d]
        avg_confidence = sum(c for d, c in buffer if d) / len(detections) if detections else 0.0
        detection_rate = len(detections) / len(buffer)
        
        # Check if gesture is stable and meets threshold
        stable_detection = (detection_rate >= 0.8 and  # 80% of frames detected
                          avg_confidence >= self.confidence_threshold)
        
        if stable_detection:
            return self._confirm_gesture(hand_label, gesture_type, avg_confidence, current_time)
        else:
            # Reset gesture start time if not stable
            if gesture_type in self.gesture_start_times[hand_label]:
                del self.gesture_start_times[hand_label][gesture_type]
        
        return None
    
    def _confirm_gesture(self, hand_label: str, gesture_type: str, confidence: float, 
                        current_time: float) -> Optional[GestureEvent]:
        """Confirm gesture after hold time and cooldown checks."""
        
        # Check cooldown
        if self._is_in_cooldown(hand_label, gesture_type, current_time):
            return None
        
        # Track gesture start time
        if gesture_type not in self.gesture_start_times[hand_label]:
            self.gesture_start_times[hand_label][gesture_type] = current_time
            return None
        
        # Check if gesture has been held long enough
        hold_duration = current_time - self.gesture_start_times[hand_label][gesture_type]
        if hold_duration < self.hold_time:
            return None
        
        # Gesture confirmed! Create event
        event = GestureEvent(
            gesture_type=gesture_type,
            hand_label=hand_label,
            confidence=confidence,
            timestamp=current_time,
            duration=hold_duration
        )
        
        # Update state
        self.current_gestures[hand_label] = gesture_type
        self.last_detection_times[hand_label][gesture_type] = current_time
        
        # Clear start time to prevent repeated detections
        del self.gesture_start_times[hand_label][gesture_type]
        
        # Clear buffer to reset detection
        self.gesture_buffers[hand_label][gesture_type] = []
        
        # Add to history
        self.history.gesture_events.append(event)
        if gesture_type not in self.history.total_detections:
            self.history.total_detections[gesture_type] = 0
        self.history.total_detections[gesture_type] += 1
        
        logger.info(f"Gesture confirmed: {gesture_type} ({hand_label} hand) - "
                   f"Confidence: {confidence:.2f}, Duration: {hold_duration:.2f}s")
        
        return event
    
    def _is_in_cooldown(self, hand_label: str, gesture_type: str, current_time: float) -> bool:
        """Check if gesture is in cooldown period."""
        if gesture_type not in self.last_detection_times[hand_label]:
            return False
        
        last_time = self.last_detection_times[hand_label][gesture_type]
        time_diff = current_time - last_time
        return time_diff < self.cooldown_time
    
    def _update_session_stats(self, events: List[GestureEvent]):
        """Update session statistics with new events."""
        if not events:
            return
        
        self.session_stats['total_gestures'] += len(events)
        
        # Calculate gestures per minute
        session_duration = time.time() - self.session_stats['session_start']
        self.session_stats['gestures_per_minute'] = (
            self.session_stats['total_gestures'] / (session_duration / 60.0)
        ) if session_duration > 0 else 0.0
        
        # Update average confidence
        all_confidences = [event.confidence for event in self.history.gesture_events]
        self.session_stats['average_confidence'] = (
            sum(all_confidences) / len(all_confidences)
        ) if all_confidences else 0.0
    
    def pause_detection(self) -> bool:
        """Pause gesture detection (triggered by stop gesture)."""
        if not self.detection_paused:
            self.detection_paused = True
            self.pause_start_time = time.time()
            logger.info("Gesture detection paused by stop gesture")
            return True
        return False
    
    def resume_detection(self) -> bool:
        """Resume gesture detection."""
        if self.detection_paused:
            self.detection_paused = False
            pause_duration = time.time() - self.pause_start_time if self.pause_start_time else 0
            self.pause_start_time = None
            logger.info(f"Gesture detection resumed after {pause_duration:.1f}s pause")
            return True
        return False
    
    def toggle_detection_pause(self) -> bool:
        """Toggle pause/resume detection state."""
        if self.detection_paused:
            return self.resume_detection()
        else:
            return self.pause_detection()
    
    def is_detection_paused(self) -> bool:
        """Check if detection is currently paused."""
        return self.detection_paused
    
    def get_current_gestures(self) -> Dict[str, Optional[str]]:
        """Get currently active gestures per hand."""
        return self.current_gestures.copy()
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        return self.session_stats.copy()
    
    def get_gesture_history(self) -> GestureHistory:
        """Get complete gesture history."""
        return self.history
    
    def clear_history(self):
        """Clear gesture history and reset counters."""
        self.history = GestureHistory()
        self.session_stats = {
            'total_gestures': 0,
            'session_start': time.time(),
            'gestures_per_minute': 0.0,
            'average_confidence': 0.0
        }
        
        # Clear all buffers and timers
        for hand_label in ['Left', 'Right']:
            self.gesture_buffers[hand_label] = {}
            self.last_detection_times[hand_label] = {}
            self.gesture_start_times[hand_label] = {}
            self.current_gestures[hand_label] = None
        
        logger.info("Gesture history and state cleared")
    
    def adjust_sensitivity(self, level: str):
        """Adjust detection sensitivity (low/medium/high)."""
        sensitivity_settings = {
            'low': {'confidence': 0.9, 'hold': 1.0, 'cooldown': 3.0},
            'medium': {'confidence': 0.8, 'hold': 0.5, 'cooldown': 2.0},
            'high': {'confidence': 0.7, 'hold': 0.3, 'cooldown': 1.5}
        }
        
        if level in sensitivity_settings:
            settings = sensitivity_settings[level]
            self.confidence_threshold = settings['confidence']
            self.hold_time = settings['hold']
            self.cooldown_time = settings['cooldown']
            
            logger.info(f"Gesture sensitivity set to {level}: "
                       f"conf={self.confidence_threshold}, "
                       f"hold={self.hold_time}s, "
                       f"cooldown={self.cooldown_time}s")
        else:
            logger.warning(f"Unknown sensitivity level: {level}")
    
    def get_cooldown_status(self) -> Dict[str, Dict[str, float]]:
        """Get remaining cooldown times for all gestures."""
        current_time = time.time()
        cooldown_status = {'Left': {}, 'Right': {}}
        
        for hand_label in ['Left', 'Right']:
            for gesture_type, last_time in self.last_detection_times[hand_label].items():
                remaining = max(0, self.cooldown_time - (current_time - last_time))
                if remaining > 0:
                    cooldown_status[hand_label][gesture_type] = remaining
        
        return cooldown_status