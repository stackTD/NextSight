"""Template Data Loader for NextSight Phase 4 Bottle Detection."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger


class TemplateLoader:
    """Load and process bottle template images for classification."""
    
    def __init__(self):
        """Initialize template loader."""
        self.project_root = Path(__file__).parent.parent.parent
        self.templates_dir = self.project_root / "templates"
        self.with_lid_dir = self.templates_dir / "with_lid"
        self.without_lid_dir = self.templates_dir / "without_lid"
        
        # Template storage
        self.with_lid_templates = []
        self.without_lid_templates = []
        self.template_features = {
            'with_lid': [],
            'without_lid': []
        }
        
        # Processing parameters
        self.target_size = (640, 480)
        self.roi_size = (200, 150)  # Region of interest for bottle neck
        
        logger.info("Template loader initialized")
        
    def load_templates(self) -> bool:
        """Load all template images from directories."""
        try:
            # Load with lid templates
            with_lid_count = self._load_category_templates(
                self.with_lid_dir, 'with_lid'
            )
            
            # Load without lid templates  
            without_lid_count = self._load_category_templates(
                self.without_lid_dir, 'without_lid'
            )
            
            if with_lid_count > 0 and without_lid_count > 0:
                logger.info(f"Templates loaded successfully: {with_lid_count} with lid, {without_lid_count} without lid")
                self._extract_features()
                return True
            else:
                logger.error("Failed to load sufficient templates")
                return False
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return False
    
    def _load_category_templates(self, directory: Path, category: str) -> int:
        """Load templates from a specific category directory."""
        count = 0
        
        if not directory.exists():
            logger.warning(f"Template directory not found: {directory}")
            return 0
        
        for image_path in directory.glob("*.jpg"):
            try:
                # Load and preprocess image
                image = cv2.imread(str(image_path))
                if image is not None:
                    processed_image = self._preprocess_template(image)
                    
                    if category == 'with_lid':
                        self.with_lid_templates.append({
                            'image': processed_image,
                            'original': image,
                            'filename': image_path.name,
                            'roi': self._extract_bottle_neck_roi(processed_image)
                        })
                    else:
                        self.without_lid_templates.append({
                            'image': processed_image,
                            'original': image,
                            'filename': image_path.name,
                            'roi': self._extract_bottle_neck_roi(processed_image)
                        })
                    
                    count += 1
                    logger.debug(f"Loaded template: {image_path.name}")
                    
            except Exception as e:
                logger.warning(f"Failed to load template {image_path}: {e}")
        
        return count
    
    def _preprocess_template(self, image: np.ndarray) -> np.ndarray:
        """Preprocess template image for feature extraction."""
        # Resize to standard size
        resized = cv2.resize(image, self.target_size)
        
        # Convert to grayscale for feature extraction
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(blurred)
        
        return enhanced
    
    def _extract_bottle_neck_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract bottle neck region of interest from template."""
        try:
            # Use the upper portion of the image where bottle necks typically are
            height, width = image.shape
            
            # Define ROI as upper-center region
            roi_x = width // 2 - self.roi_size[0] // 2
            roi_y = height // 4  # Upper quarter
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            
            roi_x_end = min(width, roi_x + self.roi_size[0])
            roi_y_end = min(height, roi_y + self.roi_size[1])
            
            roi = image[roi_y:roi_y_end, roi_x:roi_x_end]
            
            return roi
            
        except Exception as e:
            logger.warning(f"Failed to extract ROI: {e}")
            return None
    
    def _extract_features(self):
        """Extract features from all loaded templates."""
        logger.info("Extracting features from templates...")
        
        # Extract features for with_lid templates
        for template in self.with_lid_templates:
            features = self._compute_template_features(template['image'], template['roi'])
            self.template_features['with_lid'].append(features)
        
        # Extract features for without_lid templates
        for template in self.without_lid_templates:
            features = self._compute_template_features(template['image'], template['roi'])
            self.template_features['without_lid'].append(features)
        
        logger.info(f"Feature extraction complete: {len(self.template_features['with_lid'])} with lid, {len(self.template_features['without_lid'])} without lid")
    
    def _compute_template_features(self, image: np.ndarray, roi: Optional[np.ndarray]) -> Dict:
        """Compute comprehensive features for a template."""
        features = {}
        
        try:
            # Edge density features
            edges = cv2.Canny(image, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # ROI features if available
            if roi is not None:
                roi_edges = cv2.Canny(roi, 50, 150)
                features['roi_edge_density'] = np.sum(roi_edges > 0) / roi_edges.size
                
                # Circular features (caps are often circular)
                circles = cv2.HoughCircles(
                    roi, cv2.HOUGH_GRADIENT, 1, 30,
                    param1=50, param2=30, minRadius=10, maxRadius=50
                )
                features['circle_count'] = len(circles[0]) if circles is not None else 0
                
                # Intensity statistics in ROI
                features['roi_mean_intensity'] = np.mean(roi)
                features['roi_std_intensity'] = np.std(roi)
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
            logger.warning(f"Error computing features: {e}")
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
    
    def get_template_stats(self) -> Dict:
        """Get statistics about loaded templates."""
        return {
            'with_lid_count': len(self.with_lid_templates),
            'without_lid_count': len(self.without_lid_templates),
            'total_templates': len(self.with_lid_templates) + len(self.without_lid_templates),
            'target_size': self.target_size,
            'roi_size': self.roi_size,
            'features_extracted': len(self.template_features['with_lid']) > 0
        }
    
    def get_templates(self, category: str = 'all') -> Dict:
        """Get loaded templates by category."""
        if category == 'with_lid':
            return {'with_lid': self.with_lid_templates}
        elif category == 'without_lid':
            return {'without_lid': self.without_lid_templates}
        else:
            return {
                'with_lid': self.with_lid_templates,
                'without_lid': self.without_lid_templates
            }
    
    def get_template_features(self, category: str = 'all') -> Dict:
        """Get extracted features by category."""
        if category == 'with_lid':
            return {'with_lid': self.template_features['with_lid']}
        elif category == 'without_lid':
            return {'without_lid': self.template_features['without_lid']}
        else:
            return self.template_features