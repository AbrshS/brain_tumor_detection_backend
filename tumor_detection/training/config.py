from pathlib import Path

class Config:
    # Data paths
    BASE_PATH = Path("/kaggle/input/brain-tumor-mri-dataset/Data - Copy")
    TRAIN_PATH = BASE_PATH / "Training"
    TEST_PATH = BASE_PATH / "Testing"
    OUTPUT_PATH = Path("/kaggle/working/models")
    
    # Model configuration
    MODEL_NAME = "EfficientNetB7"
    IMG_SIZE = (224, 224)
    CHANNELS = 3
    NUM_CLASSES = 4
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Training hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 50
    INITIAL_LR = 1e-4
    MIN_LR = 1e-7
    LR_FACTOR = 0.5
    LR_PATIENCE = 5
    EARLY_STOPPING_PATIENCE = 10
    
    # Data augmentation parameters
    ROTATION_RANGE = 20
    ZOOM_RANGE = 0.2
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = False
    BRIGHTNESS_RANGE = [0.8, 1.2]
    
    # Mixed precision and performance
    USE_MIXED_PRECISION = True
    USE_XLA = True
    
    # Experiment tracking
    WANDB_PROJECT = "brain-tumor-classification"
    EXPERIMENT_NAME = "effnetb7_full_augmentation"
    
    # Model checkpointing
    CHECKPOINT_DIR = OUTPUT_PATH / "checkpoints"
    BEST_MODEL_PATH = OUTPUT_PATH / "best_model.h5"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True) 