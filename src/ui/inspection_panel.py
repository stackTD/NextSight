"""Quality Control Dashboard for NextSight Phase 4."""

import cv2
import numpy as np
from typing import Dict, List, Optional
import time
from loguru import logger
from detection.bottle_detector import BottleDetectionResults


class InspectionPanel:
    """Professional inspection interface for quality control."""
    
    def __init__(self):
        """Initialize inspection panel."""
        self.panel_enabled = True
        self.panel_width = 350
        self.panel_height = 200
        
        # Session tracking
        self.session_stats = {
            'session_start': time.time(),
            'total_inspected': 0,
            'total_ok': 0,
            'total_ng': 0,
            'inspection_history': [],
            'bottles_per_minute': 0.0,
            'peak_bottles_detected': 0,
            'average_confidence': 0.0
        }
        
        # Display settings
        self.bg_color = (30, 30, 30)      # Dark background
        self.text_color = (255, 255, 255)  # White text
        self.ok_color = (16, 124, 16)      # Green
        self.ng_color = (0, 0, 255)        # Red
        self.accent_color = (212, 120, 0)  # Blue accent
        self.border_color = (100, 100, 100)
        
        self.text_scale = 0.5
        self.header_scale = 0.6
        self.text_thickness = 1
        self.header_thickness = 2
        self.padding = 8
        
        logger.info("Inspection panel initialized")
    
    def render_inspection_panel(self, frame: np.ndarray, detection_results: BottleDetectionResults, 
                               detection_stats: Dict) -> np.ndarray:
        """Render the main inspection control panel."""
        if not self.panel_enabled:
            return frame
        
        try:
            # Update session statistics
            self._update_session_stats(detection_results, detection_stats)
            
            # Calculate panel position (top-left corner)
            panel_x = 10
            panel_y = 10
            
            # Draw main panel background
            cv2.rectangle(frame,
                         (panel_x, panel_y),
                         (panel_x + self.panel_width, panel_y + self.panel_height),
                         self.bg_color, -1)
            
            # Draw panel border
            cv2.rectangle(frame,
                         (panel_x, panel_y),
                         (panel_x + self.panel_width, panel_y + self.panel_height),
                         self.border_color, 2)
            
            # Render panel content
            frame = self._render_panel_header(frame, panel_x, panel_y)
            frame = self._render_real_time_stats(frame, panel_x, panel_y + 40, detection_results)
            frame = self._render_session_summary(frame, panel_x, panel_y + 100, detection_stats)
            frame = self._render_quality_metrics(frame, panel_x, panel_y + 140)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering inspection panel: {e}")
            return frame
    
    def _render_panel_header(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        """Render panel header."""
        try:
            header_text = "🏭 QUALITY CONTROL DASHBOARD"
            
            # Draw header background
            cv2.rectangle(frame,
                         (x + 2, y + 2),
                         (x + self.panel_width - 2, y + 35),
                         self.accent_color, -1)
            
            # Draw header text
            text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.header_scale, self.header_thickness)[0]
            text_x = x + (self.panel_width - text_size[0]) // 2
            
            cv2.putText(frame, header_text,
                       (text_x, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, self.header_scale, 
                       self.text_color, self.header_thickness)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering panel header: {e}")
            return frame
    
    def _render_real_time_stats(self, frame: np.ndarray, x: int, y: int, 
                               detection_results: BottleDetectionResults) -> np.ndarray:
        """Render real-time detection statistics."""
        try:
            # Current detection info
            stats_lines = [
                f"Bottles Detected: {detection_results.total_bottles}",
                f"✅ OK: {detection_results.ok_count}  ❌ NG: {detection_results.ng_count}",
                f"Avg Confidence: {detection_results.average_confidence:.1%}"
            ]
            
            current_y = y
            for i, line in enumerate(stats_lines):
                # Choose color based on content
                if "NG:" in line and detection_results.ng_count > 0:
                    line_color = self.ng_color
                elif "OK:" in line and detection_results.ok_count > 0:
                    line_color = self.ok_color
                else:
                    line_color = self.text_color
                
                cv2.putText(frame, line,
                           (x + self.padding, current_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 
                           line_color, self.text_thickness)
                current_y += 20
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering real-time stats: {e}")
            return frame
    
    def _render_session_summary(self, frame: np.ndarray, x: int, y: int, 
                               detection_stats: Dict) -> np.ndarray:
        """Render session summary statistics."""
        try:
            session_duration = time.time() - self.session_stats['session_start']
            
            summary_lines = [
                f"Session: {session_duration/60:.1f}min",
                f"Rate: {detection_stats.get('detections_per_minute', 0):.1f}/min"
            ]
            
            current_y = y
            for line in summary_lines:
                cv2.putText(frame, line,
                           (x + self.padding, current_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 
                           self.text_color, self.text_thickness)
                current_y += 20
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering session summary: {e}")
            return frame
    
    def _render_quality_metrics(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        """Render quality control metrics."""
        try:
            total_inspected = self.session_stats['total_ok'] + self.session_stats['total_ng']
            
            if total_inspected > 0:
                ok_rate = self.session_stats['total_ok'] / total_inspected
                ng_rate = self.session_stats['total_ng'] / total_inspected
                
                quality_lines = [
                    f"Total Inspected: {total_inspected}",
                    f"Quality Rate: {ok_rate:.1%} OK / {ng_rate:.1%} NG"
                ]
            else:
                quality_lines = [
                    "Total Inspected: 0",
                    "Quality Rate: -- / --"
                ]
            
            current_y = y
            for line in quality_lines:
                cv2.putText(frame, line,
                           (x + self.padding, current_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 
                           self.text_color, self.text_thickness)
                current_y += 20
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering quality metrics: {e}")
            return frame
    
    def render_bottle_details_panel(self, frame: np.ndarray, detection_results: BottleDetectionResults) -> np.ndarray:
        """Render detailed information about detected bottles."""
        if not detection_results.bottles or not self.panel_enabled:
            return frame
        
        try:
            # Position panel on the right side
            frame_height, frame_width = frame.shape[:2]
            detail_panel_width = 280
            detail_panel_x = frame_width - detail_panel_width - 10
            detail_panel_y = 10
            
            # Calculate panel height based on number of bottles
            line_height = 25
            header_height = 35
            detail_panel_height = header_height + (len(detection_results.bottles) * line_height) + 20
            
            # Draw panel background
            cv2.rectangle(frame,
                         (detail_panel_x, detail_panel_y),
                         (detail_panel_x + detail_panel_width, detail_panel_y + detail_panel_height),
                         self.bg_color, -1)
            
            # Draw panel border
            cv2.rectangle(frame,
                         (detail_panel_x, detail_panel_y),
                         (detail_panel_x + detail_panel_width, detail_panel_y + detail_panel_height),
                         self.border_color, 2)
            
            # Draw header
            header_text = "📋 BOTTLE DETAILS"
            cv2.rectangle(frame,
                         (detail_panel_x + 2, detail_panel_y + 2),
                         (detail_panel_x + detail_panel_width - 2, detail_panel_y + header_height),
                         self.accent_color, -1)
            
            text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.header_scale, self.header_thickness)[0]
            text_x = detail_panel_x + (detail_panel_width - text_size[0]) // 2
            
            cv2.putText(frame, header_text,
                       (text_x, detail_panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, self.header_scale, 
                       self.text_color, self.header_thickness)
            
            # Draw bottle details
            current_y = detail_panel_y + header_height + 10
            
            for bottle in detection_results.bottles:
                if bottle.classification:
                    status = "OK" if bottle.classification.has_cap else "NG"
                    status_color = self.ok_color if bottle.classification.has_cap else self.ng_color
                    confidence = bottle.classification.confidence
                else:
                    status = "DETECTING"
                    status_color = self.text_color
                    confidence = bottle.detection_confidence
                
                # Bottle info line
                bottle_info = f"#{bottle.bottle_id}: {status} ({confidence:.1%})"
                
                cv2.putText(frame, bottle_info,
                           (detail_panel_x + self.padding, current_y),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 
                           status_color, self.text_thickness)
                
                current_y += line_height
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering bottle details panel: {e}")
            return frame
    
    def render_alert_notifications(self, frame: np.ndarray, detection_results: BottleDetectionResults) -> np.ndarray:
        """Render alert notifications for quality control."""
        if not self.panel_enabled:
            return frame
        
        try:
            alerts = []
            
            # Check for NG bottles
            if detection_results.ng_count > 0:
                if detection_results.ng_count == 1:
                    alerts.append(("⚠️ NG Bottle Detected", self.ng_color))
                else:
                    alerts.append((f"⚠️ {detection_results.ng_count} NG Bottles Detected", self.ng_color))
            
            # Check for low confidence
            if detection_results.average_confidence < 0.7 and detection_results.total_bottles > 0:
                alerts.append(("⚠️ Low Detection Confidence", (0, 165, 255)))  # Orange
            
            # Check for no bottles
            if detection_results.detection_enabled and detection_results.total_bottles == 0:
                alerts.append(("🔍 No Bottles in View", self.text_color))
            
            # Render alerts
            if alerts:
                frame = self._render_alert_box(frame, alerts)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering alert notifications: {e}")
            return frame
    
    def _render_alert_box(self, frame: np.ndarray, alerts: List[tuple]) -> np.ndarray:
        """Render alert notification box."""
        try:
            # Position at bottom center
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate alert box size
            alert_width = 400
            alert_height = len(alerts) * 30 + 20
            alert_x = (frame_width - alert_width) // 2
            alert_y = frame_height - alert_height - 20
            
            # Draw alert background
            cv2.rectangle(frame,
                         (alert_x, alert_y),
                         (alert_x + alert_width, alert_y + alert_height),
                         self.bg_color, -1)
            
            # Draw alert border (use most severe color)
            border_color = alerts[0][1] if alerts else self.border_color
            cv2.rectangle(frame,
                         (alert_x, alert_y),
                         (alert_x + alert_width, alert_y + alert_height),
                         border_color, 2)
            
            # Draw alert text
            current_y = alert_y + 20
            for alert_text, alert_color in alerts:
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.text_scale, self.text_thickness)[0]
                text_x = alert_x + (alert_width - text_size[0]) // 2
                
                cv2.putText(frame, alert_text,
                           (text_x, current_y),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 
                           alert_color, self.text_thickness)
                current_y += 30
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering alert box: {e}")
            return frame
    
    def _update_session_stats(self, detection_results: BottleDetectionResults, detection_stats: Dict):
        """Update session statistics."""
        try:
            # Update peak bottles detected
            if detection_results.total_bottles > self.session_stats['peak_bottles_detected']:
                self.session_stats['peak_bottles_detected'] = detection_results.total_bottles
            
            # Update inspection counts
            self.session_stats['total_ok'] = detection_stats.get('ok_detections', 0)
            self.session_stats['total_ng'] = detection_stats.get('ng_detections', 0)
            self.session_stats['total_inspected'] = self.session_stats['total_ok'] + self.session_stats['total_ng']
            
            # Update bottles per minute
            self.session_stats['bottles_per_minute'] = detection_stats.get('detections_per_minute', 0)
            
            # Update average confidence
            if detection_results.total_bottles > 0:
                self.session_stats['average_confidence'] = detection_results.average_confidence
            
        except Exception as e:
            logger.error(f"Error updating session stats: {e}")
    
    def reset_session_stats(self):
        """Reset session statistics."""
        self.session_stats = {
            'session_start': time.time(),
            'total_inspected': 0,
            'total_ok': 0,
            'total_ng': 0,
            'inspection_history': [],
            'bottles_per_minute': 0.0,
            'peak_bottles_detected': 0,
            'average_confidence': 0.0
        }
        logger.info("Session statistics reset")
    
    def toggle_panel(self) -> bool:
        """Toggle inspection panel on/off."""
        self.panel_enabled = not self.panel_enabled
        status = "enabled" if self.panel_enabled else "disabled"
        logger.info(f"Inspection panel {status}")
        return self.panel_enabled
    
    def is_enabled(self) -> bool:
        """Check if panel is enabled."""
        return self.panel_enabled
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        return self.session_stats.copy()