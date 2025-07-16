"""Image processing utilities."""

import cv2
import numpy as np
from loguru import logger


class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def resize_image(image, width, height):
        """Resize image to specified dimensions."""
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def normalize_image(image):
        """Normalize image pixel values to [0, 1]."""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def crop_image(image, bbox):
        """Crop image using bounding box."""
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
        """Draw bounding box on image."""
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        return image
    
    @staticmethod
    def put_text(image, text, position, color=(255, 255, 255), font_scale=0.7):
        """Put text on image."""
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, 2)
        return image