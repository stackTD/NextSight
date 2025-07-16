"""Bottle Classification System for NextSight Phase 4."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from data.template_loader import TemplateLoader


@dataclass
class BottleClassification:
    """Data class for bottle classification results."""
    has_cap: bool
    confidence: float
    classification: str  # 'OK' or 'NG'
    template_match_score: float
    feature_similarity: float
    detection_method: str


class BottleClassifier:
    """Advanced bottle classification using template-based learning."""
    
    def __init__(self):
        """Initialize bottle classifier."""
        self.template_loader = TemplateLoader()
        self.is_initialized = False
        
        # Classification parameters
        self.confidence_threshold = 0.75
        self.template_match_threshold = 0.7
        self.feature_weights = {
            'edge_density': 0.2,
            'roi_edge_density': 0.3,
            'circle_count': 0.25,
            'roi_mean_intensity': 0.1,
            'roi_std_intensity': 0.1,
            'gradient_mean': 0.05
        }
        
        # Template matching parameters
        self.roi_size = (200, 150)
        
        logger.info("Bottle classifier initialized")
    
    def initialize(self) -> bool:
        """Initialize classifier by loading templates."""
        try:
            if self.template_loader.load_templates():
                stats = self.template_loader.get_template_stats()
                logger.info(f"Classifier initialized with {stats['total_templates']} templates")
                self.is_initialized = True
                return True
            else:
                logger.error("Failed to initialize classifier - template loading failed")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            return False
    
    def classify_bottle_region(self, image: np.ndarray, bottle_region: Tuple[int, int, int, int]) -> Optional[BottleClassification]:
        """Classify a bottle region as capped (OK) or uncapped (NG)."""
        if not self.is_initialized:
            logger.warning("Classifier not initialized")
            return None
        
        try:
            # Extract bottle region
            x, y, w, h = bottle_region
            bottle_roi = image[y:y+h, x:x+w]
            
            if bottle_roi.size == 0:
                return None
            
            # Preprocess the bottle region
            processed_roi = self._preprocess_bottle_region(bottle_roi)
            
            # Extract bottle neck ROI
            neck_roi = self._extract_bottle_neck_roi(processed_roi)
            
            # Perform classification using multiple methods
            template_result = self._template_matching_classification(processed_roi, neck_roi)
            feature_result = self._feature_based_classification(processed_roi, neck_roi)
            
            # Combine results for final classification
            final_result = self._combine_classification_results(template_result, feature_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in bottle classification: {e}")
            return None
    
    def _preprocess_bottle_region(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess bottle region for classification."""
        # Resize to standard size
        resized = cv2.resize(roi, (640, 480))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(blurred)
        
        return enhanced
    
    def _extract_bottle_neck_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract bottle neck region from processed image."""
        try:
            height, width = image.shape
            
            # Define ROI as upper-center region (where caps typically are)
            roi_x = width // 2 - self.roi_size[0] // 2
            roi_y = height // 4  # Upper quarter
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            
            roi_x_end = min(width, roi_x + self.roi_size[0])
            roi_y_end = min(height, roi_y + self.roi_size[1])
            
            neck_roi = image[roi_y:roi_y_end, roi_x:roi_x_end]
            
            return neck_roi
            
        except Exception as e:
            logger.warning(f"Failed to extract bottle neck ROI: {e}")
            return None
    
    def _template_matching_classification(self, image: np.ndarray, neck_roi: Optional[np.ndarray]) -> Dict:
        """Perform template matching classification."""
        try:
            with_lid_scores = []
            without_lid_scores = []
            
            # Get templates
            templates = self.template_loader.get_templates()
            
            # Match against with_lid templates
            for template in templates['with_lid']:
                score = self._compute_template_match_score(image, template['image'])
                with_lid_scores.append(score)
            
            # Match against without_lid templates
            for template in templates['without_lid']:
                score = self._compute_template_match_score(image, template['image'])
                without_lid_scores.append(score)
            
            # Calculate average scores
            avg_with_lid = np.mean(with_lid_scores) if with_lid_scores else 0.0
            avg_without_lid = np.mean(without_lid_scores) if without_lid_scores else 0.0
            
            # Determine classification
            if avg_with_lid > avg_without_lid:
                has_cap = True
                confidence = avg_with_lid
                template_match_score = avg_with_lid
            else:
                has_cap = False
                confidence = avg_without_lid
                template_match_score = avg_without_lid
            
            return {
                'has_cap': has_cap,
                'confidence': confidence,
                'template_match_score': template_match_score,
                'method': 'template_matching'
            }
            
        except Exception as e:
            logger.error(f"Template matching classification error: {e}")
            return {
                'has_cap': False,
                'confidence': 0.0,
                'template_match_score': 0.0,
                'method': 'template_matching'
            }
    
    def _compute_template_match_score(self, image: np.ndarray, template: np.ndarray) -> float:
        """Compute template matching score between image and template."""
        try:
            # Use normalized cross-correlation
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val
            
        except Exception as e:
            logger.warning(f"Template matching error: {e}")
            return 0.0
    
    def _feature_based_classification(self, image: np.ndarray, neck_roi: Optional[np.ndarray]) -> Dict:
        """Perform feature-based classification."""
        try:
            # Extract features from current image
            current_features = self._compute_image_features(image, neck_roi)
            
            # Get template features
            template_features = self.template_loader.get_template_features()
            
            # Compare with template features
            with_lid_similarities = []
            without_lid_similarities = []
            
            for template_feature in template_features['with_lid']:
                similarity = self._compute_feature_similarity(current_features, template_feature)
                with_lid_similarities.append(similarity)
            
            for template_feature in template_features['without_lid']:
                similarity = self._compute_feature_similarity(current_features, template_feature)
                without_lid_similarities.append(similarity)
            
            # Calculate average similarities
            avg_with_lid_sim = np.mean(with_lid_similarities) if with_lid_similarities else 0.0
            avg_without_lid_sim = np.mean(without_lid_similarities) if without_lid_similarities else 0.0
            
            # Determine classification
            if avg_with_lid_sim > avg_without_lid_sim:
                has_cap = True
                confidence = avg_with_lid_sim
                feature_similarity = avg_with_lid_sim
            else:
                has_cap = False
                confidence = avg_without_lid_sim
                feature_similarity = avg_without_lid_sim
            
            return {
                'has_cap': has_cap,
                'confidence': confidence,
                'feature_similarity': feature_similarity,
                'method': 'feature_based'
            }
            
        except Exception as e:
            logger.error(f"Feature-based classification error: {e}")
            return {
                'has_cap': False,
                'confidence': 0.0,
                'feature_similarity': 0.0,
                'method': 'feature_based'
            }
    
    def _compute_image_features(self, image: np.ndarray, neck_roi: Optional[np.ndarray]) -> Dict:
        """Compute features for the current image."""
        features = {}
        
        try:
            # Edge density features
            edges = cv2.Canny(image, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # ROI features if available
            if neck_roi is not None:
                roi_edges = cv2.Canny(neck_roi, 50, 150)
                features['roi_edge_density'] = np.sum(roi_edges > 0) / roi_edges.size
                
                # Circular features (caps are often circular)
                circles = cv2.HoughCircles(
                    neck_roi, cv2.HOUGH_GRADIENT, 1, 30,
                    param1=50, param2=30, minRadius=10, maxRadius=50
                )
                features['circle_count'] = len(circles[0]) if circles is not None else 0
                
                # Intensity statistics in ROI
                features['roi_mean_intensity'] = np.mean(neck_roi)
                features['roi_std_intensity'] = np.std(neck_roi)
            else:
                features['roi_edge_density'] = 0
                features['circle_count'] = 0
                features['roi_mean_intensity'] = 0
                features['roi_std_intensity'] = 0
            
            # Overall intensity features
            features['mean_intensity'] = np.mean(image)
            features['std_intensity'] = np.std(image)
            
            # Gradient features
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features['gradient_mean'] = np.mean(gradient_magnitude)
            
        except Exception as e:
            logger.warning(f"Error computing image features: {e}")
            # Provide default features
            features = {
                'edge_density': 0.0,
                'roi_edge_density': 0.0,
                'circle_count': 0,
                'roi_mean_intensity': 0.0,
                'roi_std_intensity': 0.0,
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'gradient_mean': 0.0
            }
        
        return features
    
    def _compute_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compute weighted similarity between two feature sets."""
        try:
            total_similarity = 0.0
            total_weight = 0.0
            
            for feature_name, weight in self.feature_weights.items():
                if feature_name in features1 and feature_name in features2:
                    val1 = features1[feature_name]
                    val2 = features2[feature_name]
                    
                    # Normalize values to prevent dominance by large values
                    if val1 == 0 and val2 == 0:
                        similarity = 1.0
                    else:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            norm_val1 = val1 / max_val
                            norm_val2 = val2 / max_val
                            similarity = 1.0 - abs(norm_val1 - norm_val2)
                        else:
                            similarity = 1.0
                    
                    total_similarity += similarity * weight
                    total_weight += weight
            
            return total_similarity / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing feature similarity: {e}")
            return 0.0
    
    def _combine_classification_results(self, template_result: Dict, feature_result: Dict) -> BottleClassification:
        """Combine template matching and feature-based results."""
        try:
            # Weighted combination of both methods
            template_weight = 0.6
            feature_weight = 0.4
            
            # Calculate combined confidence
            combined_confidence = (
                template_result['confidence'] * template_weight +
                feature_result['confidence'] * feature_weight
            )
            
            # Determine final classification based on both methods
            template_has_cap = template_result['has_cap']
            feature_has_cap = feature_result['has_cap']
            
            # If both methods agree, use that result
            if template_has_cap == feature_has_cap:
                final_has_cap = template_has_cap
            else:
                # If methods disagree, use the one with higher confidence
                if template_result['confidence'] > feature_result['confidence']:
                    final_has_cap = template_has_cap
                else:
                    final_has_cap = feature_has_cap
            
            # Ensure minimum confidence threshold
            if combined_confidence < self.confidence_threshold:
                # If confidence is too low, be conservative and mark as NG
                final_has_cap = False
                combined_confidence = max(combined_confidence, 0.5)
            
            classification = 'OK' if final_has_cap else 'NG'
            
            return BottleClassification(
                has_cap=final_has_cap,
                confidence=combined_confidence,
                classification=classification,
                template_match_score=template_result['template_match_score'],
                feature_similarity=feature_result['feature_similarity'],
                detection_method='hybrid'
            )
            
        except Exception as e:
            logger.error(f"Error combining classification results: {e}")
            return BottleClassification(
                has_cap=False,
                confidence=0.0,
                classification='NG',
                template_match_score=0.0,
                feature_similarity=0.0,
                detection_method='error'
            )
    
    def get_classifier_stats(self) -> Dict:
        """Get classifier statistics."""
        template_stats = self.template_loader.get_template_stats()
        return {
            'initialized': self.is_initialized,
            'template_stats': template_stats,
            'confidence_threshold': self.confidence_threshold,
            'template_match_threshold': self.template_match_threshold,
            'feature_weights': self.feature_weights
        }