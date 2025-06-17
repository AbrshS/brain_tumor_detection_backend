import tensorflow as tf
import os
from pathlib import Path
import logging
from datetime import datetime

from .config import Config
from .data import DataModule
from .model import BrainTumorModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup GPU and mixed precision"""
    # Enable mixed precision
    if Config.USE_MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled")
    
    # Enable XLA
    if Config.USE_XLA:
        tf.config.optimizer.set_jit(True)
        logger.info("XLA enabled")
    
    # Log GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            logger.info(f"Found GPU: {gpu}")
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("No GPUs found. Training will be slow!")

def main():
    """Main training function"""
    # Setup directories
    Config.setup_directories()
    
    # Setup GPU and performance optimizations
    setup_gpu()
    
    # Initialize data module
    logger.info("Initializing data module...")
    data_module = DataModule(Config)
    data_module.setup()
    logger.info("Data module initialized successfully")
    
    # Initialize model
    logger.info("Building model...")
    model = BrainTumorModel(Config)
    model.build()
    logger.info("Model built successfully")
    
    # Setup callbacks
    logger.info("Setting up callbacks...")
    model.setup_callbacks()
    logger.info("Callbacks setup complete")
    
    # Train model
    logger.info("Starting training...")
    try:
        history = model.train(data_module)
        logger.info("Training completed successfully")
        
        # Evaluate on test set if available
        if data_module.test_ds:
            logger.info("Evaluating on test set...")
            metrics = model.evaluate(data_module.test_ds)
            for metric_name, value in metrics.items():
                logger.info(f"Test {metric_name}: {value:.4f}")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("Training pipeline completed")

if __name__ == "__main__":
    main() 