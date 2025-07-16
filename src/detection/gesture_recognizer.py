"""Main gesture recognition engine for NextSight Phase 3."""

import time
from typing import Dict, List, Optional, Tuple
from loguru import logger

from detection.gesture_algorithms import GestureAlgorithms
from core.gesture_state import GestureState, GestureEvent
from config.settings import GESTURE_RECOGNITION_ENABLED, MAX_SIMULTANEOUS_GESTURES


class GestureRecognizer:
    """Main gesture recognition system integrating algorithms and state management."""
    
    def __init__(self):
        """Initialize the gesture recognition system."""
        self.enabled = GESTURE_RECOGNITION_ENABLED
        self.max_simultaneous = MAX_SIMULTANEOUS_GESTURES
        
        # Initialize components
        self.algorithms = GestureAlgorithms()
        self.state = GestureState()
        
        # Supported gestures
        self.supported_gestures = ['peace', 'thumbs_up', 'thumbs_down', 'ok', 'stop']
        
        # Performance tracking
        self.detection_times = []
        self.last_performance_log = time.time()
        
        logger.info("Gesture recognition engine initialized")
        logger.info(f"Supported gestures: {', '.join(self.supported_gestures)}")
        logger.info(f"Max simultaneous gestures: {self.max_simultaneous}")
    
    def process_hands(self, detection_results: Dict) -> Dict:
        """
        Process hand detection results and perform gesture recognition.
        
        Args:
            detection_results: Hand detection results from HandDetector
                             Contains: hands_detected, hands[], raw_results, etc.
        
        Returns:
            Enhanced detection results with gesture information
        """
        if not self.enabled or not detection_results['hands']:
            return self._add_empty_gesture_results(detection_results)
        
        start_time = time.time()
        
        try:
            # Perform gesture detection on each hand
            gesture_results = {}
            for hand_info in detection_results['hands']:
                hand_label = hand_info['label']
                landmarks = hand_info['landmarks']
                
                if landmarks is not None:
                    hand_gestures = self._detect_hand_gestures(landmarks, hand_label)
                    gesture_results[hand_label] = hand_gestures
            
            # Update gesture state and get confirmed events
            gesture_events = self.state.update_gesture_detection(gesture_results)
            
            # Handle special stop gesture logic
            self._handle_stop_gesture(gesture_events)
            
            # Add gesture information to detection results
            enhanced_results = self._enhance_detection_results(
                detection_results, gesture_results, gesture_events
            )
            
            # Track performance
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self._log_performance_if_needed()
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Gesture recognition error: {e}")
            return self._add_empty_gesture_results(detection_results)
    
    def _detect_hand_gestures(self, landmarks, hand_label: str) -> Dict[str, Tuple[bool, float]]:
        """Detect all gestures for a single hand."""
        gesture_results = {}
        
        # Detect each supported gesture
        for gesture_type in self.supported_gestures:
            detected, confidence = self._detect_single_gesture(landmarks, hand_label, gesture_type)
            gesture_results[gesture_type] = (detected, confidence)
        
        return gesture_results
    
    def _detect_single_gesture(self, landmarks, hand_label: str, gesture_type: str) -> Tuple[bool, float]:
        """Detect a specific gesture type."""
        try:
            if gesture_type == 'peace':
                return self.algorithms.detect_peace_sign(landmarks, hand_label)
            elif gesture_type == 'thumbs_up':
                return self.algorithms.detect_thumbs_up(landmarks, hand_label)
            elif gesture_type == 'thumbs_down':
                return self.algorithms.detect_thumbs_down(landmarks, hand_label)
            elif gesture_type == 'ok':
                return self.algorithms.detect_ok_sign(landmarks, hand_label)
            elif gesture_type == 'stop':
                return self.algorithms.detect_stop_hand(landmarks, hand_label)
            else:
                logger.warning(f"Unknown gesture type: {gesture_type}")
                return False, 0.0
                
        except Exception as e:
            logger.error(f"Error detecting {gesture_type} gesture: {e}")
            return False, 0.0
    
    def _handle_stop_gesture(self, gesture_events: List[GestureEvent]):
        """Handle special stop gesture functionality."""
        for event in gesture_events:
            if event.gesture_type == 'stop':
                # Toggle detection pause state
                self.state.toggle_detection_pause()
                logger.info(f"Stop gesture detected - Detection paused: {self.state.is_detection_paused()}")
    
    def _enhance_detection_results(self, detection_results: Dict, 
                                 gesture_results: Dict[str, Dict[str, Tuple[bool, float]]], 
                                 gesture_events: List[GestureEvent]) -> Dict:
        """Add gesture information to hand detection results."""
        enhanced = detection_results.copy()
        
        # Add gesture-specific information
        enhanced['gesture_recognition'] = {
            'enabled': self.enabled,
            'detection_paused': self.state.is_detection_paused(),
            'raw_detections': gesture_results,
            'confirmed_events': gesture_events,
            'current_gestures': self.state.get_current_gestures(),
            'session_stats': self.state.get_session_stats(),
            'cooldown_status': self.state.get_cooldown_status()
        }
        
        # Add gesture info to individual hands
        for hand_info in enhanced['hands']:
            hand_label = hand_info['label']
            
            # Add current gesture if any
            current_gesture = self.state.current_gestures.get(hand_label)
            hand_info['current_gesture'] = current_gesture
            
            # Add raw gesture detections
            if hand_label in gesture_results:
                hand_info['gesture_detections'] = gesture_results[hand_label]
            else:
                hand_info['gesture_detections'] = {}
            
            # Add gesture confidence (highest confidence among detected gestures)
            gesture_confidences = [conf for _, conf in hand_info['gesture_detections'].values()]
            hand_info['gesture_confidence'] = max(gesture_confidences) if gesture_confidences else 0.0
        
        return enhanced
    
    def _add_empty_gesture_results(self, detection_results: Dict) -> Dict:
        """Add empty gesture results when recognition is disabled or no hands detected."""
        enhanced = detection_results.copy()
        
        enhanced['gesture_recognition'] = {
            'enabled': self.enabled,
            'detection_paused': self.state.is_detection_paused(),
            'raw_detections': {},
            'confirmed_events': [],
            'current_gestures': {'Left': None, 'Right': None},
            'session_stats': self.state.get_session_stats(),
            'cooldown_status': {'Left': {}, 'Right': {}}
        }
        
        # Add empty gesture info to hands
        for hand_info in enhanced['hands']:
            hand_info['current_gesture'] = None
            hand_info['gesture_detections'] = {}
            hand_info['gesture_confidence'] = 0.0
        
        return enhanced
    
    def _log_performance_if_needed(self):
        """Log performance metrics periodically."""
        current_time = time.time()
        if current_time - self.last_performance_log > 30:  # Log every 30 seconds
            if self.detection_times:
                avg_time = sum(self.detection_times) / len(self.detection_times)
                max_time = max(self.detection_times)
                logger.info(f"Gesture recognition performance - "
                           f"Avg: {avg_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
                
                # Clear old times to prevent memory growth
                self.detection_times = self.detection_times[-100:]  # Keep last 100
            
            self.last_performance_log = current_time
    
    def toggle_recognition(self) -> bool:
        """Toggle gesture recognition on/off."""
        self.enabled = not self.enabled
        status = "enabled" if self.enabled else "disabled"
        logger.info(f"Gesture recognition {status}")
        return self.enabled
    
    def is_enabled(self) -> bool:
        """Check if gesture recognition is enabled."""
        return self.enabled
    
    def is_detection_paused(self) -> bool:
        """Check if gesture detection is currently paused."""
        return self.state.is_detection_paused()
    
    def clear_gesture_history(self):
        """Clear gesture history and reset counters."""
        self.state.clear_history()
        logger.info("Gesture history cleared")
    
    def adjust_sensitivity(self, level: str):
        """Adjust gesture detection sensitivity."""
        self.state.adjust_sensitivity(level)
        logger.info(f"Gesture sensitivity adjusted to {level}")
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        return self.state.get_session_stats()
    
    def get_gesture_history(self):
        """Get complete gesture history."""
        return self.state.get_gesture_history()
    
    def get_supported_gestures(self) -> List[str]:
        """Get list of supported gesture types."""
        return self.supported_gestures.copy()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Gesture recognition engine cleanup complete")