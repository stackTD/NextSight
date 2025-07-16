"""Bottle Detection Engine for NextSight Phase 4."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger
import time
from detection.bottle_classifier import BottleClassifier, BottleClassification


@dataclass
class BottleDetection:
    """Data class for individual bottle detection results."""
    bottle_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    classification: Optional[BottleClassification]
    detection_confidence: float
    timestamp: float = field(default_factory=time.time)
    frames_detected: int = 1


@dataclass
class BottleDetectionResults:
    """Data class for complete bottle detection results."""
    bottles: List[BottleDetection]
    total_bottles: int
    ok_count: int
    ng_count: int
    processing_time: float
    detection_enabled: bool
    average_confidence: float


class BottleDetector:
    """Robust bottle detection and cap analysis system."""
    
    def __init__(self):
        """Initialize bottle detection engine."""
        self.classifier = BottleClassifier()
        self.is_initialized = False
        self.detection_enabled = False
        
        # Detection parameters
        self.confidence_threshold = 0.75
        self.stability_requirement = 3  # Consecutive frames
        self.max_bottles = 6
        
        # Object detection parameters
        self.min_bottle_area = 5000
        self.max_bottle_area = 50000
        self.aspect_ratio_range = (0.3, 3.0)  # width/height ratio
        
        # Tracking parameters
        self.bottle_id_counter = 0
        self.tracked_bottles = {}
        self.max_tracking_distance = 100
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'ok_detections': 0,
            'ng_detections': 0,
            'average_processing_time': 0.0,
            'session_start': time.time()
        }
        
        logger.info("Bottle detection engine initialized")
    
    def initialize(self) -> bool:
        """Initialize detector by loading classifier."""
        try:
            if self.classifier.initialize():
                self.is_initialized = True
                logger.info("Bottle detector initialized successfully")
                return True
            else:
                logger.error("Failed to initialize bottle detector")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing bottle detector: {e}")
            return False
    
    def toggle_detection(self) -> bool:
        """Toggle bottle detection on/off."""
        if not self.is_initialized:
            logger.warning("Cannot toggle detection - detector not initialized")
            return False
        
        self.detection_enabled = not self.detection_enabled
        status = "enabled" if self.detection_enabled else "disabled"
        logger.info(f"Bottle detection {status}")
        return self.detection_enabled
    
    def detect_bottles(self, frame: np.ndarray) -> BottleDetectionResults:
        """Detect bottles in the current frame."""
        start_time = time.time()
        
        if not self.detection_enabled or not self.is_initialized:
            return BottleDetectionResults(
                bottles=[],
                total_bottles=0,
                ok_count=0,
                ng_count=0,
                processing_time=0.0,
                detection_enabled=self.detection_enabled,
                average_confidence=0.0
            )
        
        try:
            # Detect bottle regions
            bottle_regions = self._detect_bottle_regions(frame)
            
            # Track and classify bottles
            detected_bottles = self._track_and_classify_bottles(frame, bottle_regions)
            
            # Update tracking
            self._update_bottle_tracking(detected_bottles)
            
            # Calculate statistics
            ok_count = sum(1 for b in detected_bottles if b.classification and b.classification.has_cap)
            ng_count = sum(1 for b in detected_bottles if b.classification and not b.classification.has_cap)
            
            avg_confidence = 0.0
            if detected_bottles:
                confidences = [b.classification.confidence for b in detected_bottles if b.classification]
                avg_confidence = np.mean(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            # Update session stats
            self._update_detection_stats(detected_bottles, processing_time)
            
            return BottleDetectionResults(
                bottles=detected_bottles,
                total_bottles=len(detected_bottles),
                ok_count=ok_count,
                ng_count=ng_count,
                processing_time=processing_time,
                detection_enabled=self.detection_enabled,
                average_confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in bottle detection: {e}")
            return BottleDetectionResults(
                bottles=[],
                total_bottles=0,
                ok_count=0,
                ng_count=0,
                processing_time=time.time() - start_time,
                detection_enabled=self.detection_enabled,
                average_confidence=0.0
            )
    
    def _detect_bottle_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential bottle regions in the frame."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find bottle-like shapes
            bottle_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_bottle_area <= area <= self.max_bottle_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check aspect ratio (bottles are typically taller than wide)
                    if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                        bottle_regions.append((x, y, w, h))
            
            # Limit to max bottles and sort by area (largest first)
            bottle_regions.sort(key=lambda region: region[2] * region[3], reverse=True)
            return bottle_regions[:self.max_bottles]
            
        except Exception as e:
            logger.error(f"Error detecting bottle regions: {e}")
            return []
    
    def _track_and_classify_bottles(self, frame: np.ndarray, bottle_regions: List[Tuple[int, int, int, int]]) -> List[BottleDetection]:
        """Track bottles across frames and classify them."""
        detected_bottles = []
        
        for region in bottle_regions:
            x, y, w, h = region
            center = (x + w // 2, y + h // 2)
            
            # Try to match with existing tracked bottles
            bottle_id = self._match_tracked_bottle(center)
            
            if bottle_id is None:
                # New bottle detected
                bottle_id = self._get_next_bottle_id()
            
            # Classify the bottle
            classification = self.classifier.classify_bottle_region(frame, region)
            
            # Create detection object
            detection = BottleDetection(
                bottle_id=bottle_id,
                bbox=region,
                center=center,
                classification=classification,
                detection_confidence=classification.confidence if classification else 0.0,
                frames_detected=self.tracked_bottles.get(bottle_id, {}).get('frames_detected', 0) + 1
            )
            
            detected_bottles.append(detection)
        
        return detected_bottles
    
    def _match_tracked_bottle(self, center: Tuple[int, int]) -> Optional[int]:
        """Match detected bottle with existing tracked bottles."""
        best_match_id = None
        min_distance = float('inf')
        
        for bottle_id, tracked_info in self.tracked_bottles.items():
            if 'last_center' in tracked_info:
                distance = np.sqrt(
                    (center[0] - tracked_info['last_center'][0]) ** 2 +
                    (center[1] - tracked_info['last_center'][1]) ** 2
                )
                
                if distance < min_distance and distance < self.max_tracking_distance:
                    min_distance = distance
                    best_match_id = bottle_id
        
        return best_match_id
    
    def _get_next_bottle_id(self) -> int:
        """Get next available bottle ID."""
        self.bottle_id_counter += 1
        return self.bottle_id_counter
    
    def _update_bottle_tracking(self, detected_bottles: List[BottleDetection]):
        """Update bottle tracking information."""
        current_bottle_ids = set()
        
        for bottle in detected_bottles:
            bottle_id = bottle.bottle_id
            current_bottle_ids.add(bottle_id)
            
            self.tracked_bottles[bottle_id] = {
                'last_center': bottle.center,
                'last_detection': bottle.timestamp,
                'frames_detected': bottle.frames_detected,
                'last_classification': bottle.classification
            }
        
        # Remove bottles that haven't been seen for a while
        current_time = time.time()
        expired_bottles = []
        
        for bottle_id, tracked_info in self.tracked_bottles.items():
            if bottle_id not in current_bottle_ids:
                time_since_last = current_time - tracked_info['last_detection']
                if time_since_last > 5.0:  # 5 seconds timeout
                    expired_bottles.append(bottle_id)
        
        for bottle_id in expired_bottles:
            del self.tracked_bottles[bottle_id]
    
    def _update_detection_stats(self, detected_bottles: List[BottleDetection], processing_time: float):
        """Update detection statistics."""
        self.detection_stats['total_detections'] += len(detected_bottles)
        
        for bottle in detected_bottles:
            if bottle.classification:
                if bottle.classification.has_cap:
                    self.detection_stats['ok_detections'] += 1
                else:
                    self.detection_stats['ng_detections'] += 1
        
        # Update average processing time with exponential moving average
        alpha = 0.1
        self.detection_stats['average_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.detection_stats['average_processing_time']
        )
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics."""
        current_time = time.time()
        session_duration = current_time - self.detection_stats['session_start']
        
        return {
            'total_detections': self.detection_stats['total_detections'],
            'ok_detections': self.detection_stats['ok_detections'],
            'ng_detections': self.detection_stats['ng_detections'],
            'ok_rate': (
                self.detection_stats['ok_detections'] / 
                max(1, self.detection_stats['total_detections'])
            ),
            'ng_rate': (
                self.detection_stats['ng_detections'] / 
                max(1, self.detection_stats['total_detections'])
            ),
            'average_processing_time': self.detection_stats['average_processing_time'],
            'session_duration': session_duration,
            'detections_per_minute': (
                self.detection_stats['total_detections'] / 
                max(1, session_duration / 60)
            ),
            'tracked_bottles': len(self.tracked_bottles),
            'detection_enabled': self.detection_enabled,
            'initialized': self.is_initialized
        }
    
    def reset_detection_stats(self):
        """Reset detection statistics."""
        self.detection_stats = {
            'total_detections': 0,
            'ok_detections': 0,
            'ng_detections': 0,
            'average_processing_time': 0.0,
            'session_start': time.time()
        }
        self.tracked_bottles.clear()
        self.bottle_id_counter = 0
        logger.info("Detection statistics reset")
    
    def adjust_sensitivity(self, level: str):
        """Adjust detection sensitivity."""
        if level == 'low':
            self.confidence_threshold = 0.6
            self.min_bottle_area = 8000
            self.stability_requirement = 5
        elif level == 'medium':
            self.confidence_threshold = 0.75
            self.min_bottle_area = 5000
            self.stability_requirement = 3
        elif level == 'high':
            self.confidence_threshold = 0.85
            self.min_bottle_area = 3000
            self.stability_requirement = 2
        
        logger.info(f"Detection sensitivity set to {level}")
    
    def is_active(self) -> bool:
        """Check if detector is active."""
        return self.detection_enabled and self.is_initialized
    
    def cleanup(self):
        """Cleanup detector resources."""
        self.detection_enabled = False
        self.tracked_bottles.clear()
        logger.info("Bottle detector cleanup complete")