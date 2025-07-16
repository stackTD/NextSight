"""Professional hand detection using MediaPipe optimized for RTX 4050."""

import cv2
import numpy as np
import mediapipe as mp
from loguru import logger
from typing import List, Dict, Optional, Tuple
from config.settings import (
    HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE, 
    MAX_NUM_HANDS, HAND_MODEL_COMPLEXITY, USE_GPU_ACCELERATION
)


class HandDetector:
    """Professional MediaPipe hand detection optimized for RTX 4050."""
    
    def __init__(self):
        """Initialize MediaPipe hand detection with optimal settings."""
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure hand detection with RTX 4050 optimizations
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=HAND_MODEL_COMPLEXITY,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        
        # Hand landmark indices for finger counting
        self.finger_tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.finger_pip_ids = [3, 6, 10, 14, 18]  # PIP joints
        
        # Hand classification and tracking
        self.detection_active = True
        self.confidence_threshold = HAND_DETECTION_CONFIDENCE
        
        logger.info("MediaPipe hand detector initialized with RTX 4050 optimizations")
        logger.info(f"Configuration: max_hands={MAX_NUM_HANDS}, "
                   f"complexity={HAND_MODEL_COMPLEXITY}, "
                   f"detection_conf={HAND_DETECTION_CONFIDENCE}, "
                   f"tracking_conf={HAND_TRACKING_CONFIDENCE}")
    
    def detect_hands(self, image: np.ndarray) -> Dict:
        """
        Detect hands in the image with comprehensive results.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Dictionary containing detection results with hands info
        """
        if not self.detection_active:
            return self._empty_results()
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_image)
            
            # Initialize result structure
            detection_results = {
                'hands_detected': 0,
                'hands': [],
                'total_fingers': 0,
                'left_fingers': 0,
                'right_fingers': 0,
                'raw_results': results,
                'confidence_avg': 0.0
            }
            
            if results.multi_hand_landmarks and results.multi_handedness:
                detection_results['hands_detected'] = len(results.multi_hand_landmarks)
                
                confidences = []
                
                for idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # Determine hand side
                    hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                    hand_confidence = handedness.classification[0].score
                    confidences.append(hand_confidence)
                    
                    # Count fingers for this hand
                    fingers = self._count_fingers(hand_landmarks, hand_label)
                    
                    # Create hand info
                    hand_info = {
                        'index': idx,
                        'label': hand_label,
                        'confidence': hand_confidence,
                        'landmarks': hand_landmarks,
                        'fingers_up': fingers,
                        'finger_count': sum(fingers)
                    }
                    
                    detection_results['hands'].append(hand_info)
                    detection_results['total_fingers'] += hand_info['finger_count']
                    
                    if hand_label == 'Left':
                        detection_results['left_fingers'] = hand_info['finger_count']
                    else:
                        detection_results['right_fingers'] = hand_info['finger_count']
                
                detection_results['confidence_avg'] = np.mean(confidences)
                
                logger.debug(f"Hands detected: {detection_results['hands_detected']}, "
                           f"Total fingers: {detection_results['total_fingers']}, "
                           f"Avg confidence: {detection_results['confidence_avg']:.2f}")
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
            return self._empty_results()
    
    def _count_fingers(self, landmarks, hand_label: str) -> List[int]:
        """
        Count extended fingers for a hand.
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            List of 5 integers (0 or 1) representing finger states
        """
        fingers = []
        
        # Convert landmarks to list for easier access
        lm_list = []
        for lm in landmarks.landmark:
            lm_list.append([lm.x, lm.y])
        
        # Thumb (special case - compare with left/right direction)
        if hand_label == 'Right':
            # For right hand, thumb is up if tip is to the right of IP joint
            if lm_list[self.finger_tip_ids[0]][0] > lm_list[self.finger_tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # For left hand, thumb is up if tip is to the left of IP joint
            if lm_list[self.finger_tip_ids[0]][0] < lm_list[self.finger_tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Other fingers (compare tip with PIP joint)
        for tip_id, pip_id in zip(self.finger_tip_ids[1:], self.finger_pip_ids[1:]):
            if lm_list[tip_id][1] < lm_list[pip_id][1]:  # Tip higher than PIP
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def _empty_results(self) -> Dict:
        """Return empty detection results."""
        return {
            'hands_detected': 0,
            'hands': [],
            'total_fingers': 0,
            'left_fingers': 0,
            'right_fingers': 0,
            'raw_results': None,
            'confidence_avg': 0.0
        }
    
    def toggle_detection(self) -> bool:
        """Toggle hand detection on/off."""
        self.detection_active = not self.detection_active
        logger.info(f"Hand detection {'enabled' if self.detection_active else 'disabled'}")
        return self.detection_active
    
    def is_active(self) -> bool:
        """Check if hand detection is active."""
        return self.detection_active
    
    def get_landmark_coordinates(self, landmarks, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Convert normalized landmarks to pixel coordinates.
        
        Args:
            landmarks: MediaPipe hand landmarks
            image_shape: (height, width) of the image
            
        Returns:
            List of (x, y) pixel coordinates
        """
        height, width = image_shape[:2]
        coordinates = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            coordinates.append((x, y))
        
        return coordinates
    
    def cleanup(self):
        """Cleanup resources."""
        if self.hands:
            self.hands.close()
        logger.info("Hand detector cleanup complete")