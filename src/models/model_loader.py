"""Model loading utilities."""

import os
from pathlib import Path
import tensorflow as tf
from loguru import logger
from config.settings import MODELS_DIR


class ModelLoader:
    """Utility class for loading ML models."""
    
    @staticmethod
    def load_tensorflow_model(model_path):
        """Load TensorFlow/Keras model."""
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded TensorFlow model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    @staticmethod
    def load_tflite_model(model_path):
        """Load TensorFlow Lite model."""
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            logger.info(f"Successfully loaded TFLite model: {model_path}")
            return interpreter
        except Exception as e:
            logger.error(f"Failed to load TFLite model {model_path}: {e}")
            return None
    
    @staticmethod
    def check_model_exists(model_path):
        """Check if model file exists."""
        return Path(model_path).exists()