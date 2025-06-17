import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from pathlib import Path
import numpy as np

# Try to import wandb, but make it optional
try:
    import wandb
    from wandb.keras import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import Config

class BrainTumorModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.callbacks = []
        
    def build(self):
        """Build the EfficientNetB7 model"""
        # Base model
        base_model = EfficientNetB7(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMG_SIZE, self.config.CHANNELS)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:  # Fine-tune last 20 layers
            layer.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.INITIAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        # Create output directories if they don't exist
        self.config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Model checkpointing
        self.callbacks.extend([
            ModelCheckpoint(
                str(self.config.BEST_MODEL_PATH),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.config.CHECKPOINT_DIR / 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.3f}.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=False,
                verbose=1
            )
        ])
        
        # Early stopping and LR reduction
        self.callbacks.extend([
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.LR_FACTOR,
                patience=self.config.LR_PATIENCE,
                min_lr=self.config.MIN_LR,
                verbose=1
            )
        ])
        
        # Logging
        self.callbacks.extend([
            TensorBoard(
                log_dir=str(self.config.OUTPUT_PATH / 'logs'),
                histogram_freq=1
            ),
            CSVLogger(
                str(self.config.OUTPUT_PATH / 'training_log.csv')
            )
        ])
        
        # Add WandB callback if available
        if WANDB_AVAILABLE and hasattr(self.config, 'USE_WANDB') and self.config.USE_WANDB:
            try:
                wandb.init(
                    project=self.config.WANDB_PROJECT,
                    name=self.config.EXPERIMENT_NAME,
                    config=self.config.__dict__
                )
                self.callbacks.append(WandbCallback())
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {str(e)}")
    
    def train(self, data_module):
        """Train the model"""
        history = self.model.fit(
            data_module.train_ds,
            validation_data=data_module.val_ds,
            epochs=self.config.EPOCHS,
            callbacks=self.callbacks,
            class_weight=data_module.class_weights
        )
        return history
    
    def evaluate(self, test_ds):
        """Evaluate the model"""
        results = self.model.evaluate(test_ds)
        metrics = dict(zip(self.model.metrics_names, results))
        
        # Log to wandb if available
        if WANDB_AVAILABLE and hasattr(self.config, 'USE_WANDB') and self.config.USE_WANDB:
            try:
                wandb.log({"test_" + k: v for k, v in metrics.items()})
            except Exception as e:
                print(f"Warning: Could not log to wandb: {str(e)}")
        
        return metrics
    
    def predict(self, image):
        """Make a prediction for a single image"""
        return self.model.predict(image)
    
    def save(self, path: Path):
        """Save the model"""
        self.model.save(path)
    
    @classmethod
    def load(cls, path: Path, config: Config):
        """Load a saved model"""
        instance = cls(config)
        instance.model = tf.keras.models.load_model(path)
        return instance 