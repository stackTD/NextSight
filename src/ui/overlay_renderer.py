"""Professional overlay rendering for NextSight Phase 2."""

import cv2
import numpy as np
import time
from loguru import logger
from typing import Dict, List, Tuple, Optional
from config.settings import (
    OVERLAY_ALPHA, UI_THEME, UI_LAYOUT, HAND_COLORS,
    SHOW_PERFORMANCE_METRICS, SHOW_HAND_STATUS, 
    SHOW_FINGER_COUNT, SHOW_CONFIDENCE, OVERLAY_MODES
)


class OverlayRenderer:
    """Professional overlay renderer for NextSight with enhanced UI."""
    
    def __init__(self, alpha=OVERLAY_ALPHA):
        """Initialize professional overlay renderer."""
        self.alpha = alpha
        self.overlay_mode = 'full'  # 'full', 'minimal', 'off'
        
        # Legacy colors for backward compatibility
        self.colors = {
            'hand': (0, 255, 0),      # Green
            'object': (255, 0, 0),    # Red
            'text': (255, 255, 255),  # White
            'ok': (0, 255, 0),        # Green
            'ng': (0, 0, 255)         # Red
        }
        
        # Professional UI metrics
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.title_font = cv2.FONT_HERSHEY_DUPLEX
        
        logger.info("Professional overlay renderer initialized")
        logger.info(f"UI theme configured with {len(UI_THEME)} colors")
    
    def render_professional_ui(self, image: np.ndarray, detection_results: Dict, 
                             performance_stats: Dict) -> np.ndarray:
        """
        Render complete professional UI overlay.
        
        Args:
            image: Input frame
            detection_results: Hand detection results
            performance_stats: Performance monitoring data
            
        Returns:
            Frame with professional UI overlay
        """
        if self.overlay_mode == 'off':
            return image
        
        # Create overlay layer
        overlay = image.copy()
        
        # Render UI components based on mode
        if self.overlay_mode in ['full', 'minimal']:
            self._render_title_bar(overlay, performance_stats)
            
            if self.overlay_mode == 'full':
                self._render_status_panel(overlay, detection_results, performance_stats)
                self._render_bottom_bar(overlay, detection_results)
                
                # Render hand landmarks and overlays
                if detection_results['hands_detected'] > 0:
                    self._render_hand_landmarks(overlay, detection_results)
                    self._render_hand_info_overlay(overlay, detection_results)
        
        # Blend overlay with original image
        return cv2.addWeighted(image, 1 - self.alpha, overlay, self.alpha, 0)
    
    def _render_title_bar(self, image: np.ndarray, performance_stats: Dict):
        """Render professional title bar with branding."""
        height, width = image.shape[:2]
        bar_height = UI_LAYOUT['title_height']
        
        # Draw title background
        cv2.rectangle(image, (0, 0), (width, bar_height), UI_THEME['background'], -1)
        cv2.rectangle(image, (0, 0), (width, bar_height), UI_THEME['border'], 2)
        
        # NextSight logo/title
        title_text = "NextSight - Professional Hand Detection"
        title_pos = (UI_LAYOUT['margin'], 35)
        cv2.putText(image, title_text, title_pos, self.title_font, 
                   UI_LAYOUT['title_scale'], UI_THEME['accent'], 2)
        
        # Version and status
        version_text = "v2.0 | Phase 2"
        version_pos = (UI_LAYOUT['margin'], 50)
        cv2.putText(image, version_text, version_pos, self.font, 
                   UI_LAYOUT['text_scale']-0.1, UI_THEME['text_secondary'], 1)
        
        # Performance metrics in top right
        if SHOW_PERFORMANCE_METRICS:
            fps = performance_stats.get('current_fps', 0)
            latency = performance_stats.get('latency_ms', 0)
            
            # Choose color based on performance
            if fps >= 25:
                perf_color = UI_THEME['success']
            elif fps >= 15:
                perf_color = UI_THEME['warning']
            else:
                perf_color = UI_THEME['error']
            
            fps_text = f"FPS: {fps:.1f}"
            latency_text = f"Latency: {latency:.1f}ms"
            
            # Position from right edge
            fps_pos = (width - 180, 25)
            latency_pos = (width - 180, 45)
            
            cv2.putText(image, fps_text, fps_pos, self.font, 
                       UI_LAYOUT['text_scale'], perf_color, 1)
            cv2.putText(image, latency_text, latency_pos, self.font, 
                       UI_LAYOUT['text_scale'], perf_color, 1)
    
    def _render_status_panel(self, image: np.ndarray, detection_results: Dict, 
                           performance_stats: Dict):
        """Render status information panel."""
        height, width = image.shape[:2]
        panel_width = UI_LAYOUT['status_panel_width']
        panel_height = height - UI_LAYOUT['title_height'] - UI_LAYOUT['bottom_bar_height']
        panel_x = width - panel_width
        panel_y = UI_LAYOUT['title_height']
        
        # Draw panel background
        cv2.rectangle(image, (panel_x, panel_y), 
                     (width, panel_y + panel_height), 
                     UI_THEME['background'], -1)
        cv2.rectangle(image, (panel_x, panel_y), 
                     (width, panel_y + panel_height), 
                     UI_THEME['border'], 1)
        
        # Panel content
        margin = UI_LAYOUT['margin']
        line_height = 25
        y_pos = panel_y + margin + 20
        
        # Hand detection status
        hands_detected = detection_results['hands_detected']
        if hands_detected > 0:
            status_text = f"Hands Detected: {hands_detected}"
            status_color = UI_THEME['success']
        else:
            status_text = "No Hands Detected"
            status_color = UI_THEME['text_secondary']
        
        cv2.putText(image, status_text, (panel_x + margin, y_pos), 
                   self.font, UI_LAYOUT['text_scale'], status_color, 1)
        y_pos += line_height
        
        # Finger count display
        if SHOW_FINGER_COUNT and hands_detected > 0:
            left_fingers = detection_results['left_fingers']
            right_fingers = detection_results['right_fingers']
            
            finger_text = f"Fingers: L:{left_fingers} | R:{right_fingers}"
            cv2.putText(image, finger_text, (panel_x + margin, y_pos), 
                       self.font, UI_LAYOUT['text_scale'], UI_THEME['text_primary'], 1)
            y_pos += line_height
        
        # Detection confidence
        if SHOW_CONFIDENCE and hands_detected > 0:
            confidence = detection_results['confidence_avg']
            conf_text = f"Confidence: {confidence:.1%}"
            
            # Confidence bar
            bar_width = 200
            bar_height = 10
            bar_x = panel_x + margin
            bar_y = y_pos + 5
            
            # Background bar
            cv2.rectangle(image, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         UI_THEME['border'], -1)
            
            # Confidence fill
            fill_width = int(bar_width * confidence)
            if confidence >= 0.8:
                conf_color = UI_THEME['success']
            elif confidence >= 0.6:
                conf_color = UI_THEME['warning']
            else:
                conf_color = UI_THEME['error']
            
            cv2.rectangle(image, (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + bar_height), 
                         conf_color, -1)
            
            # Confidence text
            cv2.putText(image, conf_text, (panel_x + margin, y_pos), 
                       self.font, UI_LAYOUT['text_scale'], UI_THEME['text_primary'], 1)
            y_pos += line_height + 15
        
        # Active controls reminder
        y_pos += 10
        controls_title = "Active Controls:"
        cv2.putText(image, controls_title, (panel_x + margin, y_pos), 
                   self.font, UI_LAYOUT['text_scale'], UI_THEME['accent'], 1)
        y_pos += line_height
        
        controls = [
            "'q' - Quit", "'s' - Screenshot", "'h' - Toggle hands",
            "'o' - Toggle overlay", "'f' - Fullscreen", "'r' - Reset"
        ]
        
        for control in controls:
            cv2.putText(image, control, (panel_x + margin, y_pos), 
                       self.font, UI_LAYOUT['text_scale']-0.1, 
                       UI_THEME['text_secondary'], 1)
            y_pos += line_height - 5
    
    def _render_bottom_bar(self, image: np.ndarray, detection_results: Dict):
        """Render bottom status bar."""
        height, width = image.shape[:2]
        bar_height = UI_LAYOUT['bottom_bar_height']
        bar_y = height - bar_height
        
        # Draw background
        cv2.rectangle(image, (0, bar_y), (width, height), UI_THEME['background'], -1)
        cv2.rectangle(image, (0, bar_y), (width, height), UI_THEME['border'], 2)
        
        # Status indicators
        margin = UI_LAYOUT['margin']
        
        # Hand presence indicator
        if detection_results['hands_detected'] > 0:
            hand_status = "👋 Hands Active"
            hand_color = UI_THEME['success']
        else:
            hand_status = "🚫 No Hands"
            hand_color = UI_THEME['text_secondary']
        
        cv2.putText(image, hand_status, (margin, bar_y + 30), 
                   self.font, UI_LAYOUT['header_scale'], hand_color, 2)
        
        # Performance status
        perf_status = "Performance: Excellent ✅"  # Simplified for now
        cv2.putText(image, perf_status, (margin, bar_y + 60), 
                   self.font, UI_LAYOUT['text_scale'], UI_THEME['success'], 1)
    
    def _render_hand_landmarks(self, image: np.ndarray, detection_results: Dict):
        """Render hand landmarks with professional styling."""
        if not detection_results['hands']:
            return
        
        for hand_info in detection_results['hands']:
            landmarks = hand_info['landmarks']
            hand_label = hand_info['label']
            confidence = hand_info['confidence']
            
            # Skip if no landmarks (mock mode)
            if landmarks is None:
                continue
            
            # Choose color based on hand
            if hand_label == 'Right':
                hand_color = HAND_COLORS['right_hand']
            else:
                hand_color = HAND_COLORS['left_hand']
            
            # Draw landmarks
            height, width = image.shape[:2]
            for idx, landmark in enumerate(landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Larger circles for fingertips
                if idx in [4, 8, 12, 16, 20]:  # Fingertips
                    radius = int(8 * confidence)
                    color = HAND_COLORS['fingertips']
                else:
                    radius = int(5 * confidence)
                    color = hand_color
                
                cv2.circle(image, (x, y), radius, color, -1)
                cv2.circle(image, (x, y), radius + 1, UI_THEME['text_primary'], 1)
            
            # Draw hand connections (skeleton)
            self._draw_hand_connections(image, landmarks, hand_color, confidence)
    
    def _draw_hand_connections(self, image: np.ndarray, landmarks, color: Tuple, confidence: float):
        """Draw hand skeleton connections."""
        # Skip if no landmarks (mock mode)
        if landmarks is None:
            return
            
        height, width = image.shape[:2]
        thickness = max(1, int(3 * confidence))
        
        # Hand connection patterns (MediaPipe standard)
        connections = [
            # Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            # Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            # Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            # Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            # Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
            # Palm
            [5, 9], [9, 13], [13, 17]
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            start_pos = (int(start_point.x * width), int(start_point.y * height))
            end_pos = (int(end_point.x * width), int(end_point.y * height))
            
            cv2.line(image, start_pos, end_pos, color, thickness)
    
    def _render_hand_info_overlay(self, image: np.ndarray, detection_results: Dict):
        """Render real-time hand information overlay."""
        if not detection_results['hands']:
            return
        
        y_offset = 100  # Start below title bar
        
        for hand_info in detection_results['hands']:
            hand_label = hand_info['label']
            finger_count = hand_info['finger_count']
            confidence = hand_info['confidence']
            
            # Create info text
            if hand_label == 'Left':
                emoji = "👆"
                color = HAND_COLORS['left_hand']
            else:
                emoji = "✋"
                color = HAND_COLORS['right_hand']
            
            info_text = f"{hand_label}: {emoji}{finger_count}"
            
            # Draw with background
            text_size = cv2.getTextSize(info_text, self.font, 
                                      UI_LAYOUT['header_scale'], 2)[0]
            
            # Background rectangle
            bg_margin = 5
            cv2.rectangle(image, 
                         (10 - bg_margin, y_offset - text_size[1] - bg_margin),
                         (10 + text_size[0] + bg_margin, y_offset + bg_margin),
                         UI_THEME['background'], -1)
            
            # Text
            cv2.putText(image, info_text, (10, y_offset), self.font,
                       UI_LAYOUT['header_scale'], color, 2)
            
            y_offset += 40
    
    def set_overlay_mode(self, mode: str) -> str:
        """Set overlay display mode."""
        if mode in OVERLAY_MODES:
            self.overlay_mode = mode
            logger.info(f"Overlay mode set to: {mode}")
        return self.overlay_mode
    
    def cycle_overlay_mode(self) -> str:
        """Cycle through overlay modes."""
        current_idx = OVERLAY_MODES.index(self.overlay_mode)
        next_idx = (current_idx + 1) % len(OVERLAY_MODES)
        self.overlay_mode = OVERLAY_MODES[next_idx]
        logger.info(f"Overlay mode cycled to: {self.overlay_mode}")
        return self.overlay_mode
    
    # Legacy methods for backward compatibility
    def draw_hand_landmarks(self, image, landmarks):
        """Legacy method - use render_professional_ui instead."""
        logger.warning("Using legacy draw_hand_landmarks - consider upgrading to render_professional_ui")
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
        """Legacy FPS drawing - integrated into professional UI."""
        pass
    
    def draw_status(self, image, status_text, position=(10, 60)):
        """Legacy status drawing - integrated into professional UI."""
        pass