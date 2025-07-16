#!/usr/bin/env python3
"""Demonstration script for NextSight Phase 4 bottle detection system."""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from core.bottle_controls import BottleControls
from data.template_loader import TemplateLoader


def create_demo_frame_with_bottles():
    """Create a demo frame with simulated bottles for testing."""
    # Create a base frame (simulated camera feed)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add a gradient background
    for y in range(720):
        for x in range(1280):
            frame[y, x] = [50 + (x * 50) // 1280, 80 + (y * 40) // 720, 60]
    
    # Simulate bottle shapes by drawing rectangles and ellipses
    bottle_regions = [
        (200, 150, 120, 300),  # Bottle 1
        (400, 200, 100, 250),  # Bottle 2
        (650, 100, 110, 350),  # Bottle 3
        (900, 180, 130, 280),  # Bottle 4
    ]
    
    for i, (x, y, w, h) in enumerate(bottle_regions):
        # Draw bottle body
        bottle_color = (120 + i * 20, 150 + i * 15, 180 + i * 10)
        cv2.rectangle(frame, (x, y), (x + w, y + h), bottle_color, -1)
        
        # Add some texture/pattern
        cv2.rectangle(frame, (x + 10, y + 10), (x + w - 10, y + h - 10), 
                     (bottle_color[0] + 30, bottle_color[1] + 30, bottle_color[2] + 30), 2)
        
        # Simulate bottle neck (top portion)
        neck_w = w // 3
        neck_h = h // 4
        neck_x = x + (w - neck_w) // 2
        neck_y = y
        
        cv2.rectangle(frame, (neck_x, neck_y), (neck_x + neck_w, neck_y + neck_h), 
                     (bottle_color[0] + 50, bottle_color[1] + 50, bottle_color[2] + 50), -1)
        
        # Add bottle number
        cv2.putText(frame, f"B{i+1}", (x + w//2 - 15, y + h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame


def demonstrate_bottle_detection():
    """Demonstrate bottle detection functionality."""
    logger.info("Starting NextSight Phase 4 bottle detection demonstration...")
    
    # Initialize bottle controls
    bottle_controls = BottleControls()
    
    if not bottle_controls.initialize():
        logger.error("Failed to initialize bottle controls")
        return False
    
    logger.info("✅ Bottle controls initialized successfully")
    
    # Load template information
    template_loader = TemplateLoader()
    if template_loader.load_templates():
        stats = template_loader.get_template_stats()
        logger.info(f"✅ Loaded {stats['total_templates']} template images")
        logger.info(f"   With lid: {stats['with_lid_count']}, Without lid: {stats['without_lid_count']}")
    
    # Create demo frames
    logger.info("Creating demonstration frames...")
    
    # Demo frame 1: Basic detection test
    demo_frame = create_demo_frame_with_bottles()
    
    # Enable bottle detection
    bottle_controls._toggle_bottle_detection()
    logger.info("✅ Bottle detection enabled")
    
    # Process the demo frame
    detection_results, detection_stats = bottle_controls.process_bottle_detection(demo_frame)
    
    if detection_results:
        logger.info(f"✅ Detection results: {detection_results.total_bottles} bottles detected")
        logger.info(f"   Processing time: {detection_results.processing_time*1000:.1f}ms")
    
    # Render the UI
    demo_frame_with_ui = bottle_controls.render_bottle_ui(demo_frame, detection_results, detection_stats)
    
    # Save demonstration screenshot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_filename = f"nextsight_phase4_demo_{timestamp}.jpg"
    
    cv2.imwrite(screenshot_filename, demo_frame_with_ui)
    logger.info(f"✅ Demo screenshot saved: {screenshot_filename}")
    
    # Test different sensitivity levels
    logger.info("Testing sensitivity levels...")
    
    for sensitivity in ['low', 'medium', 'high']:
        bottle_controls._adjust_detection_sensitivity()
        logger.info(f"✅ Sensitivity set to {bottle_controls.sensitivity_level}")
    
    # Test overlay toggle
    logger.info("Testing overlay controls...")
    bottle_controls._toggle_inspection_overlay()
    bottle_controls._toggle_inspection_overlay()  # Toggle back on
    logger.info("✅ Overlay controls tested")
    
    # Generate final demo frame with all features
    final_demo_frame = create_demo_frame_with_bottles()
    
    # Add some simulated noise/variation
    noise = np.random.randint(-20, 20, final_demo_frame.shape, dtype=np.int16)
    final_demo_frame = np.clip(final_demo_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Process final frame
    final_results, final_stats = bottle_controls.process_bottle_detection(final_demo_frame)
    final_frame_with_ui = bottle_controls.render_bottle_ui(final_demo_frame, final_results, final_stats)
    
    # Save final screenshot
    final_screenshot = f"nextsight_phase4_final_demo_{timestamp}.jpg"
    cv2.imwrite(final_screenshot, final_frame_with_ui)
    logger.info(f"✅ Final demo screenshot saved: {final_screenshot}")
    
    # Print control status
    status = bottle_controls.get_control_status()
    logger.info("Control Status:")
    logger.info(f"   Detection enabled: {status['detection_enabled']}")
    logger.info(f"   Overlay enabled: {status['overlay_enabled']}")
    logger.info(f"   Panel enabled: {status['panel_enabled']}")
    logger.info(f"   Sensitivity level: {status['sensitivity_level']}")
    
    # Print help text
    help_text = bottle_controls.get_help_text()
    logger.info("Control Help:")
    for line in help_text.split('\n'):
        logger.info(f"   {line}")
    
    # Cleanup
    bottle_controls.cleanup()
    logger.info("✅ Cleanup complete")
    
    logger.info("🎉 NextSight Phase 4 demonstration complete!")
    logger.info("🚀 Key features demonstrated:")
    logger.info("   - Template-based bottle classification")
    logger.info("   - Real-time detection with OK/NG status")
    logger.info("   - Professional UI overlays")
    logger.info("   - Quality control dashboard")
    logger.info("   - Multiple sensitivity levels")
    logger.info("   - Comprehensive statistics tracking")
    
    return True


def test_template_processing():
    """Test template image processing and feature extraction."""
    logger.info("Testing template image processing...")
    
    template_loader = TemplateLoader()
    
    if not template_loader.load_templates():
        logger.error("Failed to load templates")
        return False
    
    # Get template statistics
    stats = template_loader.get_template_stats()
    logger.info(f"Template Statistics:")
    logger.info(f"   With lid templates: {stats['with_lid_count']}")
    logger.info(f"   Without lid templates: {stats['without_lid_count']}")
    logger.info(f"   Total templates: {stats['total_templates']}")
    logger.info(f"   Target size: {stats['target_size']}")
    logger.info(f"   ROI size: {stats['roi_size']}")
    
    # Get feature information
    features = template_loader.get_template_features()
    
    if features['with_lid'] and features['without_lid']:
        sample_with_lid = features['with_lid'][0]
        sample_without_lid = features['without_lid'][0]
        
        logger.info("Feature Analysis:")
        logger.info(f"   Feature types: {list(sample_with_lid.keys())}")
        
        # Compare average features
        avg_with_lid = {}
        avg_without_lid = {}
        
        for feature_name in sample_with_lid.keys():
            avg_with_lid[feature_name] = np.mean([f[feature_name] for f in features['with_lid']])
            avg_without_lid[feature_name] = np.mean([f[feature_name] for f in features['without_lid']])
        
        logger.info("Average Features Comparison:")
        for feature_name in avg_with_lid.keys():
            with_val = avg_with_lid[feature_name]
            without_val = avg_without_lid[feature_name]
            diff = abs(with_val - without_val)
            logger.info(f"   {feature_name}: With={with_val:.3f}, Without={without_val:.3f}, Diff={diff:.3f}")
    
    return True


if __name__ == "__main__":
    logger.info("Starting NextSight Phase 4 demonstration script...")
    
    # Test template processing
    if test_template_processing():
        logger.info("✅ Template processing test passed")
    else:
        logger.error("❌ Template processing test failed")
        sys.exit(1)
    
    # Demonstrate bottle detection
    if demonstrate_bottle_detection():
        logger.info("✅ Bottle detection demonstration completed successfully")
    else:
        logger.error("❌ Bottle detection demonstration failed")
        sys.exit(1)
    
    logger.info("🎊 All demonstrations completed successfully!")
    logger.info("NextSight Phase 4 is ready for professional bottle inspection!")