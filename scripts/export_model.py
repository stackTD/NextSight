"""Model export utilities."""

import sys
from pathlib import Path
import tensorflow as tf
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config.settings import MODELS_DIR


def export_to_tflite(model_path, output_path):
    """Export Keras model to TensorFlow Lite."""
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"Model exported to TFLite: {output_path}")
        
    except Exception as e:
        logger.error(f"TFLite export failed: {e}")
        raise


def export_to_onnx(model_path, output_path):
    """Export model to ONNX format."""
    # TODO: Implement ONNX export
    logger.info(f"ONNX export not implemented yet")
    pass


def main():
    """Main export function."""
    logger.info("Starting model export...")
    
    try:
        model_path = MODELS_DIR / "jar_classifier.h5"
        tflite_path = MODELS_DIR / "jar_classifier.tflite"
        
        if model_path.exists():
            export_to_tflite(model_path, tflite_path)
        else:
            logger.warning(f"Model not found: {model_path}")
        
        logger.info("Model export completed!")
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise


if __name__ == "__main__":
    main()