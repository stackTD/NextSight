"""Overlay rendering for visualization."""

import cv2
import numpy as np
from loguru import logger
from config.settings import OVERLAY_ALPHA


class OverlayRenderer:
    """Renders overlays for hands, objects, and information."""
    
    def __init__(self, alpha=OVERLAY_ALPHA):
        """Initialize overlay renderer."""
        self.alpha = alpha
        self.colors = {
            'hand': (0, 255, 0),      # Green
            'object': (255, 0, 0),    # Red
            'text': (255, 255, 255),  # White
            'ok': (0, 255, 0),        # Green
            'ng': (0, 0, 255)         # Red
        }
        logger.info("Overlay renderer initialized")
    
    def draw_hand_landmarks(self, image, landmarks):
        """Draw hand landmarks on the image."""
        # TODO: Implement hand landmark drawing
        pass
    
    def draw_object_bbox(self, image, bbox, label, confidence):
        """Draw object bounding box with label."""
        x, y, w, h = bbox
        color = self.colors['ok'] if 'lid' in label.lower() else self.colors['ng']
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label_text = f"{label} ({confidence:.2f})"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Draw label text
        cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.colors['text'], 2)
    
    def draw_fps(self, image, fps):
        """Draw FPS counter on the image."""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   self.colors['text'], 2)
    
    def draw_status(self, image, status_text, position=(10, 60)):
        """Draw status text on the image."""
        cv2.putText(image, status_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.colors['text'], 2)