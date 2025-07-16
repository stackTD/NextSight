"""Advanced gesture detection algorithms for NextSight Phase 3."""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from loguru import logger
from config.settings import GESTURE_DETECTION_PARAMS


class GestureAlgorithms:
    """Precise gesture detection algorithms using MediaPipe landmarks."""
    
    def __init__(self):
        """Initialize gesture detection algorithms."""
        # MediaPipe hand landmark indices
        self.finger_tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.finger_pip_ids = [3, 6, 10, 14, 18]  # PIP joints
        self.finger_mcp_ids = [2, 5, 9, 13, 17]   # MCP joints
        self.wrist_id = 0
        
        # Gesture parameters
        self.params = GESTURE_DETECTION_PARAMS
        
        logger.info("Gesture algorithms initialized with precision detection")
    
    def detect_peace_sign(self, landmarks, hand_label: str) -> Tuple[bool, float]:
        """
        Detect peace sign gesture (✌️).
        Algorithm: Check index and middle fingers extended, others closed.
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            (is_peace, confidence_score)
        """
        try:
            lm_list = self._landmarks_to_list(landmarks)
            
            # Check if index and middle fingers are extended
            index_extended = self._is_finger_extended(lm_list, 1, hand_label)  # Index finger
            middle_extended = self._is_finger_extended(lm_list, 2, hand_label)  # Middle finger
            
            # Check if thumb, ring, and pinky are closed
            thumb_closed = not self._is_finger_extended(lm_list, 0, hand_label)
            ring_closed = not self._is_finger_extended(lm_list, 3, hand_label)
            pinky_closed = not self._is_finger_extended(lm_list, 4, hand_label)
            
            # Check separation between index and middle fingers
            index_tip = lm_list[self.finger_tip_ids[1]]
            middle_tip = lm_list[self.finger_tip_ids[2]]
            separation = self._calculate_distance(index_tip, middle_tip)
            
            # Peace sign criteria
            peace_fingers = index_extended and middle_extended
            closed_fingers = thumb_closed and ring_closed and pinky_closed
            good_separation = separation > self.params['peace']['tip_distance_threshold']
            
            # Calculate confidence based on finger positions
            confidence = 0.0
            if peace_fingers:
                confidence += 0.4
            if closed_fingers:
                confidence += 0.4
            if good_separation:
                confidence += 0.2
            
            is_peace = peace_fingers and closed_fingers and good_separation
            
            return is_peace, confidence
            
        except Exception as e:
            logger.error(f"Peace sign detection error: {e}")
            return False, 0.0
    
    def detect_thumbs_up(self, landmarks, hand_label: str) -> Tuple[bool, float]:
        """
        Detect thumbs up gesture (👍).
        Algorithm: Thumb orientation upward, other fingers closed.
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            (is_thumbs_up, confidence_score)
        """
        try:
            lm_list = self._landmarks_to_list(landmarks)
            
            # Check thumb orientation (should point upward)
            thumb_up = self._is_thumb_pointing_up(lm_list, hand_label)
            
            # Check if other fingers are closed/curled
            fingers_closed = 0
            for finger_idx in range(1, 5):  # Index, Middle, Ring, Pinky
                if not self._is_finger_extended(lm_list, finger_idx, hand_label):
                    fingers_closed += 1
            
            # Check thumb extension
            thumb_extended = self._is_finger_extended(lm_list, 0, hand_label)
            
            # Thumbs up criteria
            good_thumb = thumb_up and thumb_extended
            good_fingers = fingers_closed >= 3  # At least 3 fingers closed
            
            # Calculate confidence
            confidence = 0.0
            if thumb_up:
                confidence += 0.3
            if thumb_extended:
                confidence += 0.2
            confidence += (fingers_closed / 4.0) * 0.5  # Based on closed fingers
            
            is_thumbs_up = good_thumb and good_fingers
            
            return is_thumbs_up, confidence
            
        except Exception as e:
            logger.error(f"Thumbs up detection error: {e}")
            return False, 0.0
    
    def detect_thumbs_down(self, landmarks, hand_label: str) -> Tuple[bool, float]:
        """
        Detect thumbs down gesture (👎).
        Algorithm: Thumb orientation downward, other fingers closed.
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            (is_thumbs_down, confidence_score)
        """
        try:
            lm_list = self._landmarks_to_list(landmarks)
            
            # Check thumb orientation (should point downward)
            thumb_down = self._is_thumb_pointing_down(lm_list, hand_label)
            
            # Check if other fingers are closed/curled
            fingers_closed = 0
            for finger_idx in range(1, 5):  # Index, Middle, Ring, Pinky
                if not self._is_finger_extended(lm_list, finger_idx, hand_label):
                    fingers_closed += 1
            
            # Check thumb extension
            thumb_extended = self._is_finger_extended(lm_list, 0, hand_label)
            
            # Thumbs down criteria
            good_thumb = thumb_down and thumb_extended
            good_fingers = fingers_closed >= 3  # At least 3 fingers closed
            
            # Calculate confidence
            confidence = 0.0
            if thumb_down:
                confidence += 0.3
            if thumb_extended:
                confidence += 0.2
            confidence += (fingers_closed / 4.0) * 0.5  # Based on closed fingers
            
            is_thumbs_down = good_thumb and good_fingers
            
            return is_thumbs_down, confidence
            
        except Exception as e:
            logger.error(f"Thumbs down detection error: {e}")
            return False, 0.0
    
    def detect_ok_sign(self, landmarks, hand_label: str) -> Tuple[bool, float]:
        """
        Detect OK gesture (👌).
        Algorithm: Thumb-index circle formation, other fingers extended.
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            (is_ok, confidence_score)
        """
        try:
            lm_list = self._landmarks_to_list(landmarks)
            
            # Check circle formation between thumb and index finger
            thumb_tip = lm_list[self.finger_tip_ids[0]]
            index_tip = lm_list[self.finger_tip_ids[1]]
            circle_distance = self._calculate_distance(thumb_tip, index_tip)
            
            # Check if other fingers are extended
            middle_extended = self._is_finger_extended(lm_list, 2, hand_label)
            ring_extended = self._is_finger_extended(lm_list, 3, hand_label)
            pinky_extended = self._is_finger_extended(lm_list, 4, hand_label)
            
            # OK sign criteria
            good_circle = circle_distance < self.params['ok']['circle_distance_threshold']
            extended_fingers = sum([middle_extended, ring_extended, pinky_extended]) >= 2
            
            # Calculate confidence
            confidence = 0.0
            if good_circle:
                confidence += 0.5
            confidence += (sum([middle_extended, ring_extended, pinky_extended]) / 3.0) * 0.5
            
            is_ok = good_circle and extended_fingers
            
            return is_ok, confidence
            
        except Exception as e:
            logger.error(f"OK sign detection error: {e}")
            return False, 0.0
    
    def detect_stop_hand(self, landmarks, hand_label: str) -> Tuple[bool, float]:
        """
        Detect stop gesture (✋).
        Algorithm: All fingers extended and spread apart.
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            (is_stop, confidence_score)
        """
        try:
            lm_list = self._landmarks_to_list(landmarks)
            
            # Check if all fingers are extended
            all_extended = True
            extended_count = 0
            for finger_idx in range(5):
                if self._is_finger_extended(lm_list, finger_idx, hand_label):
                    extended_count += 1
                else:
                    all_extended = False
            
            # Check finger spread (angles between adjacent fingers)
            finger_spread = self._calculate_finger_spread(lm_list)
            good_spread = finger_spread > self.params['stop']['spread_angle_threshold']
            
            # Check palm orientation (should face camera)
            palm_facing = self._is_palm_facing_camera(lm_list)
            
            # Stop hand criteria
            good_extension = extended_count >= 4  # At least 4 fingers extended
            
            # Calculate confidence
            confidence = (extended_count / 5.0) * 0.6
            if good_spread:
                confidence += 0.2
            if palm_facing:
                confidence += 0.2
            
            is_stop = good_extension and good_spread
            
            return is_stop, confidence
            
        except Exception as e:
            logger.error(f"Stop hand detection error: {e}")
            return False, 0.0
    
    def _landmarks_to_list(self, landmarks) -> List[List[float]]:
        """Convert landmarks to list format."""
        lm_list = []
        for lm in landmarks.landmark:
            lm_list.append([lm.x, lm.y])
        return lm_list
    
    def _is_finger_extended(self, lm_list: List[List[float]], finger_idx: int, hand_label: str) -> bool:
        """Check if a finger is extended."""
        if finger_idx == 0:  # Thumb
            return self._is_thumb_extended(lm_list, hand_label)
        else:
            # For other fingers, compare tip with PIP joint
            tip_y = lm_list[self.finger_tip_ids[finger_idx]][1]
            pip_y = lm_list[self.finger_pip_ids[finger_idx]][1]
            return tip_y < pip_y  # Tip is higher than PIP
    
    def _is_thumb_extended(self, lm_list: List[List[float]], hand_label: str) -> bool:
        """Check if thumb is extended."""
        thumb_tip = lm_list[self.finger_tip_ids[0]]
        thumb_ip = lm_list[self.finger_tip_ids[0] - 1]
        
        if hand_label == 'Right':
            return thumb_tip[0] > thumb_ip[0]  # Tip is to the right of IP
        else:
            return thumb_tip[0] < thumb_ip[0]  # Tip is to the left of IP
    
    def _is_thumb_pointing_up(self, lm_list: List[List[float]], hand_label: str) -> bool:
        """Check if thumb is pointing upward."""
        thumb_tip = lm_list[self.finger_tip_ids[0]]
        thumb_mcp = lm_list[self.finger_mcp_ids[0]]
        
        # Calculate vertical displacement
        vertical_diff = thumb_mcp[1] - thumb_tip[1]  # Positive if thumb points up
        return vertical_diff > 0.05  # Threshold for upward pointing
    
    def _is_thumb_pointing_down(self, lm_list: List[List[float]], hand_label: str) -> bool:
        """Check if thumb is pointing downward."""
        thumb_tip = lm_list[self.finger_tip_ids[0]]
        thumb_mcp = lm_list[self.finger_mcp_ids[0]]
        
        # Calculate vertical displacement
        vertical_diff = thumb_tip[1] - thumb_mcp[1]  # Positive if thumb points down
        return vertical_diff > 0.05  # Threshold for downward pointing
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _calculate_finger_spread(self, lm_list: List[List[float]]) -> float:
        """Calculate the spread angle between fingers."""
        # Calculate angles between adjacent finger tips
        wrist = lm_list[self.wrist_id]
        angles = []
        
        for i in range(len(self.finger_tip_ids) - 1):
            tip1 = lm_list[self.finger_tip_ids[i]]
            tip2 = lm_list[self.finger_tip_ids[i + 1]]
            
            # Calculate angles from wrist to each fingertip
            angle1 = math.atan2(tip1[1] - wrist[1], tip1[0] - wrist[0])
            angle2 = math.atan2(tip2[1] - wrist[1], tip2[0] - wrist[0])
            
            # Calculate angle difference
            angle_diff = abs(angle1 - angle2)
            angles.append(math.degrees(angle_diff))
        
        # Return average spread angle
        return sum(angles) / len(angles) if angles else 0.0
    
    def _is_palm_facing_camera(self, lm_list: List[List[float]]) -> bool:
        """Check if palm is facing the camera (simplified)."""
        # Use the relative positions of landmarks to estimate palm orientation
        # This is a simplified check - in reality, this would need more sophisticated 3D analysis
        wrist = lm_list[self.wrist_id]
        middle_mcp = lm_list[self.finger_mcp_ids[2]]
        
        # If MCP is above wrist, likely palm is facing camera
        return middle_mcp[1] < wrist[1]