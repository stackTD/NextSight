"""Bottle Visual Feedback System for NextSight Phase 4."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from loguru import logger
from detection.bottle_detector import BottleDetectionResults, BottleDetection
from config.settings import UI_THEME


class BottleOverlay:
    """Professional bottle inspection display system."""
    
    def __init__(self):
        """Initialize bottle overlay renderer."""
        self.overlay_enabled = True
        
        # Visual styling
        self.ok_color = (16, 124, 16)      # Green for OK bottles (BGR)
        self.ng_color = (0, 0, 255)        # Red for NG bottles (BGR)
        self.detection_color = (100, 100, 100)  # Gray for unclassified
        self.text_color = (255, 255, 255)  # White text
        self.bg_color = (0, 0, 0)          # Black background for text
        
        # Styling parameters
        self.rectangle_thickness = 3
        self.corner_radius = 10
        self.text_scale = 0.6
        self.text_thickness = 2
        self.padding = 5
        
        # Animation parameters
        self.animation_duration = 0.5
        self.fade_animations = {}
        
        logger.info("Bottle overlay system initialized")
    
    def render_bottle_overlays(self, frame: np.ndarray, detection_results: BottleDetectionResults) -> np.ndarray:
        """Render bottle detection overlays on frame."""
        if not self.overlay_enabled or not detection_results.detection_enabled:
            return frame
        
        try:
            overlay_frame = frame.copy()
            
            # Render individual bottles
            for bottle in detection_results.bottles:
                overlay_frame = self._render_bottle_overlay(overlay_frame, bottle)
            
            # Render status summary if bottles detected
            if detection_results.total_bottles > 0:
                overlay_frame = self._render_detection_summary(overlay_frame, detection_results)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Error rendering bottle overlays: {e}")
            return frame
    
    def _render_bottle_overlay(self, frame: np.ndarray, bottle: BottleDetection) -> np.ndarray:
        """Render overlay for individual bottle."""
        try:
            x, y, w, h = bottle.bbox
            
            # Determine color and status based on classification
            if bottle.classification is None:
                color = self.detection_color
                status = "DETECTING..."
                confidence = bottle.detection_confidence
            elif bottle.classification.has_cap:
                color = self.ok_color
                status = "OK ✅"
                confidence = bottle.classification.confidence
            else:
                color = self.ng_color
                status = "NG ❌"
                confidence = bottle.classification.confidence
            
            # Draw rounded rectangle
            frame = self._draw_rounded_rectangle(frame, (x, y, w, h), color, self.rectangle_thickness)
            
            # Prepare status text
            bottle_label = f"Bottle #{bottle.bottle_id}: {status}"
            confidence_text = f"Confidence: {confidence:.1%}"
            
            # Calculate text positions
            label_size = cv2.getTextSize(bottle_label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)[0]
            conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)[0]
            
            # Position text above the bottle
            text_bg_width = max(label_size[0], conf_size[0]) + 2 * self.padding
            text_bg_height = label_size[1] + conf_size[1] + 3 * self.padding
            
            text_x = x
            text_y = max(text_bg_height, y - 5)
            
            # Draw text background
            cv2.rectangle(frame, 
                         (text_x, text_y - text_bg_height),
                         (text_x + text_bg_width, text_y),
                         self.bg_color, -1)
            
            # Draw text border
            cv2.rectangle(frame,
                         (text_x, text_y - text_bg_height),
                         (text_x + text_bg_width, text_y),
                         color, 2)
            
            # Draw status text
            cv2.putText(frame, bottle_label,
                       (text_x + self.padding, text_y - conf_size[1] - 2 * self.padding),
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_color, self.text_thickness)
            
            # Draw confidence text
            cv2.putText(frame, confidence_text,
                       (text_x + self.padding, text_y - self.padding),
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_color, self.text_thickness)
            
            # Add frame count indicator for stable detection
            if bottle.frames_detected >= 3:
                stability_indicator = "●"  # Solid circle for stable
                stability_color = color
            else:
                stability_indicator = "○"  # Empty circle for unstable
                stability_color = (128, 128, 128)
            
            cv2.putText(frame, stability_indicator,
                       (x + w - 20, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, stability_color, 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering bottle overlay: {e}")
            return frame
    
    def _draw_rounded_rectangle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                               color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        """Draw a rounded rectangle around the bottle."""
        try:
            x, y, w, h = bbox
            
            # Draw main rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw rounded corners (simplified version)
            corner_size = min(self.corner_radius, w // 4, h // 4)
            
            # Top-left corner
            cv2.ellipse(frame, (x + corner_size, y + corner_size), 
                       (corner_size, corner_size), 180, 0, 90, color, thickness)
            
            # Top-right corner
            cv2.ellipse(frame, (x + w - corner_size, y + corner_size),
                       (corner_size, corner_size), 270, 0, 90, color, thickness)
            
            # Bottom-left corner
            cv2.ellipse(frame, (x + corner_size, y + h - corner_size),
                       (corner_size, corner_size), 90, 0, 90, color, thickness)
            
            # Bottom-right corner
            cv2.ellipse(frame, (x + w - corner_size, y + h - corner_size),
                       (corner_size, corner_size), 0, 0, 90, color, thickness)
            
            return frame
            
        except Exception as e:
            logger.warning(f"Error drawing rounded rectangle: {e}")
            # Fallback to regular rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            return frame
    
    def _render_detection_summary(self, frame: np.ndarray, detection_results: BottleDetectionResults) -> np.ndarray:
        """Render detection summary information."""
        try:
            # Prepare summary text
            summary_lines = [
                f"Bottles Detected: {detection_results.total_bottles}",
                f"✅ OK: {detection_results.ok_count} | ❌ NG: {detection_results.ng_count}",
                f"Overall Confidence: {detection_results.average_confidence:.1%}",
                f"Processing: {detection_results.processing_time*1000:.1f}ms"
            ]
            
            # Calculate background size
            max_text_width = 0
            total_height = 0
            line_heights = []
            
            for line in summary_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)[0]
                max_text_width = max(max_text_width, text_size[0])
                line_heights.append(text_size[1])
                total_height += text_size[1] + self.padding
            
            # Position summary in top-right corner
            frame_height, frame_width = frame.shape[:2]
            bg_width = max_text_width + 2 * self.padding
            bg_height = total_height + self.padding
            
            bg_x = frame_width - bg_width - 10
            bg_y = 10
            
            # Draw background
            cv2.rectangle(frame,
                         (bg_x, bg_y),
                         (bg_x + bg_width, bg_y + bg_height),
                         self.bg_color, -1)
            
            # Draw border with appropriate color
            border_color = self.ng_color if detection_results.ng_count > 0 else self.ok_color
            cv2.rectangle(frame,
                         (bg_x, bg_y),
                         (bg_x + bg_width, bg_y + bg_height),
                         border_color, 2)
            
            # Draw summary text
            current_y = bg_y + self.padding
            for i, line in enumerate(summary_lines):
                current_y += line_heights[i]
                cv2.putText(frame, line,
                           (bg_x + self.padding, current_y),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_color, self.text_thickness)
                current_y += self.padding
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering detection summary: {e}")
            return frame
    
    def render_inspection_status(self, frame: np.ndarray, detection_enabled: bool) -> np.ndarray:
        """Render bottle inspection status indicator."""
        try:
            # Status text
            if detection_enabled:
                status_text = "🔍 BOTTLE INSPECTION: ACTIVE"
                status_color = self.ok_color
            else:
                status_text = "🔍 BOTTLE INSPECTION: INACTIVE"
                status_color = self.ng_color
            
            # Calculate text size and position
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.text_scale, self.text_thickness)[0]
            
            # Position at bottom-left
            bg_x = 10
            bg_y = frame.shape[0] - text_size[1] - 2 * self.padding - 10
            bg_width = text_size[0] + 2 * self.padding
            bg_height = text_size[1] + 2 * self.padding
            
            # Draw background
            cv2.rectangle(frame,
                         (bg_x, bg_y),
                         (bg_x + bg_width, bg_y + bg_height),
                         self.bg_color, -1)
            
            # Draw border
            cv2.rectangle(frame,
                         (bg_x, bg_y),
                         (bg_x + bg_width, bg_y + bg_height),
                         status_color, 2)
            
            # Draw status text
            cv2.putText(frame, status_text,
                       (bg_x + self.padding, bg_y + text_size[1] + self.padding),
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_color, self.text_thickness)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering inspection status: {e}")
            return frame
    
    def render_alert_overlay(self, frame: np.ndarray, ng_count: int) -> np.ndarray:
        """Render alert overlay for NG detections."""
        if ng_count == 0:
            return frame
        
        try:
            # Flash red border for multiple NG bottles
            if ng_count > 1:
                current_time = time.time()
                flash_intensity = int(127 * (1 + np.sin(current_time * 10))) + 128
                
                # Draw flashing border
                border_color = (0, 0, flash_intensity)
                frame_height, frame_width = frame.shape[:2]
                
                cv2.rectangle(frame, (0, 0), (frame_width-1, frame_height-1), border_color, 5)
                
                # Alert text
                alert_text = f"⚠️ MULTIPLE NG BOTTLES: {ng_count}"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                text_x = (frame_width - text_size[0]) // 2
                text_y = 50
                
                # Draw alert text with background
                cv2.rectangle(frame,
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                
                cv2.putText(frame, alert_text,
                           (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering alert overlay: {e}")
            return frame
    
    def toggle_overlay(self) -> bool:
        """Toggle overlay display on/off."""
        self.overlay_enabled = not self.overlay_enabled
        status = "enabled" if self.overlay_enabled else "disabled"
        logger.info(f"Bottle overlay {status}")
        return self.overlay_enabled
    
    def is_enabled(self) -> bool:
        """Check if overlay is enabled."""
        return self.overlay_enabled