"""Enhanced Detection Controls for NextSight Phase 4."""

import time
from typing import Dict, Optional
from loguru import logger
from detection.bottle_detector import BottleDetector
from ui.bottle_overlay import BottleOverlay
from ui.inspection_panel import InspectionPanel


class BottleControls:
    """Intelligent bottle inspection management system."""
    
    def __init__(self):
        """Initialize bottle controls."""
        self.bottle_detector = BottleDetector()
        self.bottle_overlay = BottleOverlay()
        self.inspection_panel = InspectionPanel()
        
        # Control state
        self.detection_enabled = False
        self.overlay_enabled = True
        self.panel_enabled = True
        self.sensitivity_level = 'medium'
        self.auto_save_enabled = False
        
        # Screenshot management
        self.screenshot_counter = 0
        
        logger.info("Bottle controls initialized")
    
    def initialize(self) -> bool:
        """Initialize all bottle detection components."""
        try:
            if self.bottle_detector.initialize():
                logger.info("Bottle controls initialized successfully")
                return True
            else:
                logger.error("Failed to initialize bottle controls")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing bottle controls: {e}")
            return False
    
    def handle_keyboard_input(self, key: int, frame=None) -> bool:
        """Handle bottle detection keyboard controls. Returns True if key was handled."""
        try:
            if key == ord('b'):
                return self._toggle_bottle_detection()
            elif key == ord('i'):
                return self._toggle_inspection_overlay()
            elif key == ord('s') and frame is not None:
                return self._take_inspection_screenshot(frame)
            elif key == ord('r'):
                return self._reset_inspection_counters()
            elif key == ord('t'):
                return self._adjust_detection_sensitivity()
            else:
                return False  # Key not handled by bottle controls
                
        except Exception as e:
            logger.error(f"Error handling bottle control input: {e}")
            return False
    
    def _toggle_bottle_detection(self) -> bool:
        """Toggle bottle detection on/off."""
        try:
            self.detection_enabled = self.bottle_detector.toggle_detection()
            status = "enabled" if self.detection_enabled else "disabled"
            logger.info(f"Bottle detection {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error toggling bottle detection: {e}")
            return False
    
    def _toggle_inspection_overlay(self) -> bool:
        """Toggle inspection overlay display."""
        try:
            self.overlay_enabled = self.bottle_overlay.toggle_overlay()
            self.panel_enabled = self.inspection_panel.toggle_panel()
            
            status = "enabled" if self.overlay_enabled else "disabled"
            logger.info(f"Inspection overlay {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error toggling inspection overlay: {e}")
            return False
    
    def _take_inspection_screenshot(self, frame) -> bool:
        """Take inspection screenshot with current results."""
        try:
            self.screenshot_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"nextsight_phase4_inspection_{timestamp}_{self.screenshot_counter:03d}.jpg"
            
            # Save screenshot
            import cv2
            cv2.imwrite(filename, frame)
            
            logger.info(f"Inspection screenshot saved: {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error taking inspection screenshot: {e}")
            return False
    
    def _reset_inspection_counters(self) -> bool:
        """Reset inspection counters and statistics."""
        try:
            self.bottle_detector.reset_detection_stats()
            self.inspection_panel.reset_session_stats()
            
            logger.info("Inspection counters reset")
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting inspection counters: {e}")
            return False
    
    def _adjust_detection_sensitivity(self) -> bool:
        """Cycle through detection sensitivity levels."""
        try:
            sensitivity_levels = ['low', 'medium', 'high']
            current_index = sensitivity_levels.index(self.sensitivity_level)
            next_index = (current_index + 1) % len(sensitivity_levels)
            self.sensitivity_level = sensitivity_levels[next_index]
            
            # Apply sensitivity to detector
            self.bottle_detector.adjust_sensitivity(self.sensitivity_level)
            
            logger.info(f"Detection sensitivity set to {self.sensitivity_level}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting detection sensitivity: {e}")
            return False
    
    def process_bottle_detection(self, frame):
        """Process bottle detection on current frame."""
        try:
            # Perform bottle detection
            detection_results = self.bottle_detector.detect_bottles(frame)
            
            # Get detection statistics
            detection_stats = self.bottle_detector.get_detection_stats()
            
            return detection_results, detection_stats
            
        except Exception as e:
            logger.error(f"Error processing bottle detection: {e}")
            return None, None
    
    def render_bottle_ui(self, frame, detection_results, detection_stats):
        """Render complete bottle detection UI."""
        try:
            # Render bottle overlays
            if self.overlay_enabled and detection_results:
                frame = self.bottle_overlay.render_bottle_overlays(frame, detection_results)
                frame = self.bottle_overlay.render_inspection_status(frame, self.detection_enabled)
                
                # Render alert overlay for NG detections
                if detection_results.ng_count > 0:
                    frame = self.bottle_overlay.render_alert_overlay(frame, detection_results.ng_count)
            
            # Render inspection panels
            if self.panel_enabled and detection_results and detection_stats:
                frame = self.inspection_panel.render_inspection_panel(frame, detection_results, detection_stats)
                frame = self.inspection_panel.render_bottle_details_panel(frame, detection_results)
                frame = self.inspection_panel.render_alert_notifications(frame, detection_results)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering bottle UI: {e}")
            return frame
    
    def get_control_status(self) -> Dict:
        """Get current control status."""
        return {
            'detection_enabled': self.detection_enabled,
            'overlay_enabled': self.overlay_enabled,
            'panel_enabled': self.panel_enabled,
            'sensitivity_level': self.sensitivity_level,
            'auto_save_enabled': self.auto_save_enabled,
            'screenshot_counter': self.screenshot_counter,
            'detector_initialized': self.bottle_detector.is_initialized,
            'detector_active': self.bottle_detector.is_active()
        }
    
    def get_help_text(self) -> str:
        """Get help text for bottle detection controls."""
        return (
            "Bottle Detection Controls:\n"
            "'b' = Toggle bottle detection on/off\n"
            "'i' = Toggle inspection overlay display\n"
            "'s' = Take inspection screenshot\n"
            "'r' = Reset inspection counters\n"
            "'t' = Adjust detection sensitivity (low/medium/high)"
        )
    
    def cleanup(self):
        """Cleanup bottle control resources."""
        try:
            if self.bottle_detector:
                self.bottle_detector.cleanup()
            
            logger.info("Bottle controls cleanup complete")
            
        except Exception as e:
            logger.error(f"Error in bottle controls cleanup: {e}")
    
    def is_detection_active(self) -> bool:
        """Check if bottle detection is currently active."""
        return self.detection_enabled and self.bottle_detector.is_active()
    
    def get_detection_stats(self) -> Optional[Dict]:
        """Get current detection statistics."""
        try:
            if self.bottle_detector.is_initialized:
                return self.bottle_detector.get_detection_stats()
            return None
            
        except Exception as e:
            logger.error(f"Error getting detection stats: {e}")
            return None