"""Interactive message overlay system for NextSight Phase 3."""

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from config.settings import (
    GESTURE_MESSAGES, GESTURE_MESSAGE_DURATION, 
    GESTURE_ANIMATION_DURATION, UI_THEME
)


@dataclass
class MessageInfo:
    """Information about a displayed message."""
    text: str
    color: Tuple[int, int, int]
    start_time: float
    duration: float
    position: Tuple[int, int]
    font_size: float = 1.0
    confidence: float = 0.0
    hand_label: str = ""


class MessageOverlay:
    """Professional interactive message display system."""
    
    def __init__(self):
        """Initialize the message overlay system."""
        self.enabled = True
        self.message_duration = GESTURE_MESSAGE_DURATION
        self.animation_duration = GESTURE_ANIMATION_DURATION
        
        # Active messages
        self.active_messages: List[MessageInfo] = []
        self.message_queue: List[MessageInfo] = []
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.base_font_size = 1.2
        self.font_thickness = 3
        self.shadow_offset = (3, 3)
        self.shadow_color = (0, 0, 0)  # Black shadow
        
        # Message positioning
        self.center_position = None  # Will be set based on frame size
        self.line_spacing = 60
        self.max_simultaneous_messages = 3
        
        # Animation settings
        self.fade_in_frames = int(30 * self.animation_duration)  # Assuming 30 FPS
        self.fade_out_frames = int(30 * self.animation_duration)
        
        logger.info("Interactive message overlay system initialized")
        logger.info(f"Message duration: {self.message_duration}s, "
                   f"Animation: {self.animation_duration}s")
    
    def add_gesture_message(self, gesture_type: str, hand_label: str, confidence: float):
        """
        Add a gesture message to the display queue.
        
        Args:
            gesture_type: Type of gesture detected
            hand_label: Which hand detected the gesture
            confidence: Detection confidence (0.0-1.0)
        """
        if not self.enabled or gesture_type not in GESTURE_MESSAGES:
            return
        
        message_config = GESTURE_MESSAGES[gesture_type]
        current_time = time.time()
        
        # Create message info
        message = MessageInfo(
            text=message_config['text'],
            color=message_config['color'],
            start_time=current_time,
            duration=self.message_duration,
            position=(0, 0),  # Will be calculated during rendering
            font_size=self.base_font_size,
            confidence=confidence,
            hand_label=hand_label
        )
        
        # Add to queue
        self.message_queue.append(message)
        
        logger.info(f"Gesture message queued: {gesture_type} ({hand_label}) - {message.text}")
    
    def render_messages(self, image: np.ndarray) -> np.ndarray:
        """
        Render all active messages on the image.
        
        Args:
            image: Input image to render messages on
            
        Returns:
            Image with messages rendered
        """
        if not self.enabled:
            return image
        
        # Update center position based on image size
        height, width = image.shape[:2]
        self.center_position = (width // 2, height // 2)
        
        # Process message queue
        self._process_message_queue()
        
        # Update active messages
        self._update_active_messages()
        
        # Render all active messages
        for i, message in enumerate(self.active_messages):
            image = self._render_single_message(image, message, i)
        
        return image
    
    def _process_message_queue(self):
        """Process queued messages and add them to active list."""
        while (self.message_queue and 
               len(self.active_messages) < self.max_simultaneous_messages):
            message = self.message_queue.pop(0)
            self.active_messages.append(message)
            logger.debug(f"Message activated: {message.text}")
    
    def _update_active_messages(self):
        """Update active messages and remove expired ones."""
        current_time = time.time()
        
        # Remove expired messages
        self.active_messages = [
            msg for msg in self.active_messages
            if current_time - msg.start_time < msg.duration + self.animation_duration
        ]
    
    def _render_single_message(self, image: np.ndarray, message: MessageInfo, index: int) -> np.ndarray:
        """Render a single message with animations and effects."""
        current_time = time.time()
        message_age = current_time - message.start_time
        
        # Calculate alpha for fade in/out animation
        alpha = self._calculate_message_alpha(message_age, message.duration)
        if alpha <= 0:
            return image
        
        # Calculate position
        position = self._calculate_message_position(image.shape, index)
        
        # Create message text with confidence and hand info
        main_text = message.text
        info_text = f"{message.hand_label} Hand - Confidence: {message.confidence:.0%}"
        
        # Render main message
        image = self._render_text_with_effects(
            image, main_text, position, message.color, 
            message.font_size, alpha
        )
        
        # Render info text (smaller, below main text)
        info_position = (position[0], position[1] + 40)
        image = self._render_text_with_effects(
            image, info_text, info_position, UI_THEME['text_secondary'],
            message.font_size * 0.6, alpha * 0.8
        )
        
        return image
    
    def _calculate_message_alpha(self, message_age: float, duration: float) -> float:
        """Calculate alpha value for fade in/out animation."""
        fade_in_time = self.animation_duration
        fade_out_start = duration - self.animation_duration
        
        if message_age < fade_in_time:
            # Fade in
            return message_age / fade_in_time
        elif message_age > fade_out_start:
            # Fade out
            fade_progress = (message_age - fade_out_start) / self.animation_duration
            return max(0, 1.0 - fade_progress)
        else:
            # Fully visible
            return 1.0
    
    def _calculate_message_position(self, image_shape: Tuple[int, int, int], index: int) -> Tuple[int, int]:
        """Calculate position for message based on index."""
        height, width = image_shape[:2]
        
        # Center horizontally, stack vertically
        center_x = width // 2
        
        # Start from center and spread vertically
        base_y = height // 2
        offset_y = (index - len(self.active_messages) // 2) * self.line_spacing
        
        return (center_x, base_y + offset_y)
    
    def _render_text_with_effects(self, image: np.ndarray, text: str, 
                                 position: Tuple[int, int], color: Tuple[int, int, int],
                                 font_size: float, alpha: float) -> np.ndarray:
        """Render text with shadow effects and alpha blending."""
        
        # Calculate text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_size, self.font_thickness
        )
        
        # Center the text
        text_x = position[0] - text_width // 2
        text_y = position[1] + text_height // 2
        
        # Create overlay for alpha blending
        overlay = image.copy()
        
        # Draw shadow
        shadow_x = text_x + self.shadow_offset[0]
        shadow_y = text_y + self.shadow_offset[1]
        cv2.putText(overlay, text, (shadow_x, shadow_y), 
                   self.font, font_size, self.shadow_color, self.font_thickness + 1)
        
        # Draw main text
        cv2.putText(overlay, text, (text_x, text_y), 
                   self.font, font_size, color, self.font_thickness)
        
        # Apply alpha blending
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image
    
    def add_custom_message(self, text: str, color: Tuple[int, int, int] = None, 
                          duration: float = None):
        """
        Add a custom message to the display.
        
        Args:
            text: Message text to display
            color: Text color (BGR), defaults to accent color
            duration: Display duration in seconds
        """
        if not self.enabled:
            return
        
        if color is None:
            color = UI_THEME['accent']
        if duration is None:
            duration = self.message_duration
        
        message = MessageInfo(
            text=text,
            color=color,
            start_time=time.time(),
            duration=duration,
            position=(0, 0),
            font_size=self.base_font_size * 0.8  # Slightly smaller for custom messages
        )
        
        self.message_queue.append(message)
        logger.info(f"Custom message queued: {text}")
    
    def clear_messages(self):
        """Clear all active and queued messages."""
        self.active_messages.clear()
        self.message_queue.clear()
        logger.info("All messages cleared")
    
    def toggle_messages(self) -> bool:
        """Toggle message display on/off."""
        self.enabled = not self.enabled
        status = "enabled" if self.enabled else "disabled"
        logger.info(f"Message overlay {status}")
        
        if not self.enabled:
            self.clear_messages()
        
        return self.enabled
    
    def is_enabled(self) -> bool:
        """Check if message overlay is enabled."""
        return self.enabled
    
    def set_message_duration(self, duration: float):
        """Set default message display duration."""
        self.message_duration = duration
        logger.info(f"Message duration set to {duration}s")
    
    def get_active_message_count(self) -> int:
        """Get number of currently active messages."""
        return len(self.active_messages)
    
    def get_queued_message_count(self) -> int:
        """Get number of queued messages."""
        return len(self.message_queue)
    
    def render_gesture_status(self, image: np.ndarray, gesture_info: Dict) -> np.ndarray:
        """
        Render gesture status information overlay.
        
        Args:
            image: Input image
            gesture_info: Gesture recognition information
            
        Returns:
            Image with gesture status overlay
        """
        if not self.enabled or not gesture_info.get('enabled', False):
            return image
        
        height, width = image.shape[:2]
        
        # Status panel position (top-left corner)
        panel_x = 10
        panel_y = 80  # Below title bar
        line_height = 25
        
        # Gesture recognition status
        if gesture_info.get('detection_paused', False):
            status_text = "🔴 Gesture Detection: PAUSED"
            status_color = UI_THEME['error']
        else:
            status_text = "🟢 Gesture Detection: ACTIVE"
            status_color = UI_THEME['success']
        
        cv2.putText(image, status_text, (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Current gestures
        current_gestures = gesture_info.get('current_gestures', {})
        y_pos = panel_y + line_height
        
        for hand_label, gesture in current_gestures.items():
            if gesture:
                gesture_text = f"{hand_label}: {gesture} ✓"
                color = UI_THEME['success']
            else:
                gesture_text = f"{hand_label}: No gesture"
                color = UI_THEME['text_secondary']
            
            cv2.putText(image, gesture_text, (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += line_height
        
        # Session statistics
        stats = gesture_info.get('session_stats', {})
        total = stats.get('total_gestures', 0)
        avg_conf = stats.get('average_confidence', 0)
        
        stats_text = f"Total: {total} | Avg Confidence: {avg_conf:.1%}"
        cv2.putText(image, stats_text, (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_THEME['text_primary'], 1)
        
        return image