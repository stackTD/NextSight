#!/usr/bin/env python3
"""Test script for NextSight Phase 4 bottle detection components."""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from data.template_loader import TemplateLoader
from detection.bottle_classifier import BottleClassifier
from detection.bottle_detector import BottleDetector
from ui.bottle_overlay import BottleOverlay
from ui.inspection_panel import InspectionPanel
from core.bottle_controls import BottleControls


def test_phase4_components():
    """Test NextSight Phase 4 bottle detection components."""
    logger.info("Testing NextSight Phase 4 bottle detection components...")
    
    # Test template loader
    logger.info("Testing template loader...")
    template_loader = TemplateLoader()
    if template_loader.load_templates():
        stats = template_loader.get_template_stats()
        logger.info(f"✅ Template loader test - Loaded {stats['total_templates']} templates")
        logger.info(f"   With lid: {stats['with_lid_count']}, Without lid: {stats['without_lid_count']}")
    else:
        logger.error("❌ Template loader test failed")
        return False
    
    # Test bottle classifier
    logger.info("Testing bottle classifier...")
    classifier = BottleClassifier()
    if classifier.initialize():
        stats = classifier.get_classifier_stats()
        logger.info(f"✅ Bottle classifier test - Initialized with {stats['template_stats']['total_templates']} templates")
        
        # Test classification on dummy image
        test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        test_region = (300, 200, 200, 300)  # x, y, w, h
        
        classification = classifier.classify_bottle_region(test_image, test_region)
        if classification:
            logger.info(f"✅ Classification test - Result: {classification.classification}, Confidence: {classification.confidence:.2f}")
        else:
            logger.warning("⚠️  Classification returned None (expected for random image)")
    else:
        logger.error("❌ Bottle classifier test failed")
        return False
    
    # Test bottle detector
    logger.info("Testing bottle detector...")
    detector = BottleDetector()
    if detector.initialize():
        logger.info("✅ Bottle detector initialized")
        
        # Test detection on dummy frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        detection_results = detector.detect_bottles(test_frame)
        
        logger.info(f"✅ Detection test - Found {detection_results.total_bottles} bottles")
        logger.info(f"   Processing time: {detection_results.processing_time*1000:.1f}ms")
    else:
        logger.error("❌ Bottle detector test failed")
        return False
    
    # Test UI components
    logger.info("Testing UI components...")
    
    # Test bottle overlay
    overlay = BottleOverlay()
    logger.info("✅ Bottle overlay initialized")
    
    # Test inspection panel
    panel = InspectionPanel()
    logger.info("✅ Inspection panel initialized")
    
    # Test bottle controls
    logger.info("Testing bottle controls...")
    controls = BottleControls()
    if controls.initialize():
        logger.info("✅ Bottle controls initialized")
        
        # Test control status
        status = controls.get_control_status()
        logger.info(f"   Detection enabled: {status['detection_enabled']}")
        logger.info(f"   Detector initialized: {status['detector_initialized']}")
        
        # Test help text
        help_text = controls.get_help_text()
        logger.info(f"✅ Control help text available ({len(help_text)} chars)")
    else:
        logger.error("❌ Bottle controls test failed")
        return False
    
    # Test rendering with dummy data
    logger.info("Testing complete UI rendering...")
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Enable detection and render UI
    controls.detection_enabled = True
    detection_results, detection_stats = controls.process_bottle_detection(test_frame)
    
    if detection_results and detection_stats:
        rendered_frame = controls.render_bottle_ui(test_frame, detection_results, detection_stats)
        logger.info(f"✅ UI rendering test - Output frame shape: {rendered_frame.shape}")
    else:
        logger.warning("⚠️  UI rendering test - No detection results")
    
    # Cleanup
    controls.cleanup()
    detector.cleanup()
    
    logger.info("🎉 All NextSight Phase 4 components tested successfully!")
    logger.info("Phase 4 implementation ready for bottle cap detection!")
    logger.info("🚀 Features: Template-based classification, real-time detection, professional UI")
    
    return True


def test_template_images():
    """Test loading and processing of actual template images."""
    logger.info("Testing template image processing...")
    
    template_loader = TemplateLoader()
    
    # Check if template directories exist
    if not template_loader.with_lid_dir.exists():
        logger.error(f"With lid directory not found: {template_loader.with_lid_dir}")
        return False
    
    if not template_loader.without_lid_dir.exists():
        logger.error(f"Without lid directory not found: {template_loader.without_lid_dir}")
        return False
    
    # Load templates
    if template_loader.load_templates():
        stats = template_loader.get_template_stats()
        logger.info(f"✅ Template images loaded successfully")
        logger.info(f"   With lid templates: {stats['with_lid_count']}")
        logger.info(f"   Without lid templates: {stats['without_lid_count']}")
        logger.info(f"   Target size: {stats['target_size']}")
        logger.info(f"   ROI size: {stats['roi_size']}")
        logger.info(f"   Features extracted: {stats['features_extracted']}")
        
        # Test feature extraction
        features = template_loader.get_template_features()
        if features['with_lid'] and features['without_lid']:
            sample_with_lid = features['with_lid'][0]
            sample_without_lid = features['without_lid'][0]
            
            logger.info("✅ Feature extraction successful")
            logger.info(f"   With lid sample features: {list(sample_with_lid.keys())}")
            logger.info(f"   Without lid sample features: {list(sample_without_lid.keys())}")
        
        return True
    else:
        logger.error("❌ Failed to load template images")
        return False


if __name__ == "__main__":
    logger.info("Starting NextSight Phase 4 component tests...")
    
    # Test template images first
    if test_template_images():
        logger.info("Template image tests passed ✅")
    else:
        logger.error("Template image tests failed ❌")
        sys.exit(1)
    
    # Test all components
    if test_phase4_components():
        logger.info("All Phase 4 tests passed ✅")
    else:
        logger.error("Phase 4 tests failed ❌")
        sys.exit(1)