import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split

# Configuration class
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
    
    # Wandb configuration (optional)
    USE_WANDB = False  # Set to True if you want to use wandb
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

class DataModule:
    def __init__(self, config: Config):
        self.config = config
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_weights = None
    
    def setup(self):
        """Setup all datasets and compute class weights"""
        # Load and split data
        images, labels = self._load_data(self.config.TRAIN_PATH)
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Compute class weights
        self.class_weights = self._compute_class_weights(train_labels)
        
        # Create datasets
        self.train_ds = self._create_dataset(train_imgs, train_labels, is_training=True)
        self.val_ds = self._create_dataset(val_imgs, val_labels, is_training=False)
        
        # Load test data if available
        if self.config.TEST_PATH.exists():
            test_imgs, test_labels = self._load_data(self.config.TEST_PATH)
            self.test_ds = self._create_dataset(test_imgs, test_labels, is_training=False)
    
    def _load_data(self, data_path: Path) -> Tuple[List[str], List[int]]:
        """Load image paths and labels from directory"""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.config.CLASS_NAMES):
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob("*.jpg"):
                images.append(str(img_path))
                labels.append(class_idx)
        
        return images, labels
    
    def _compute_class_weights(self, labels: List[int]) -> dict:
        """Compute balanced class weights"""
        total = len(labels)
        class_counts = np.bincount(labels)
        class_weights = {i: total / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts)}
        return class_weights
    
    def _preprocess_image(self, img_path: str) -> tf.Tensor:
        """Load and preprocess a single image"""
        # Read image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Convert to grayscale and back to RGB
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)
        
        # Resize and normalize
        img = tf.image.resize(img, self.config.IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        
        return img
    
    def _augment(self, image: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation"""
        # Random flip
        if self.config.HORIZONTAL_FLIP:
            image = tf.image.random_flip_left_right(image)
        if self.config.VERTICAL_FLIP:
            image = tf.image.random_flip_up_down(image)
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random rotation
        angle = tf.random.uniform([], -self.config.ROTATION_RANGE, self.config.ROTATION_RANGE) * np.pi / 180
        image = tf.keras.layers.experimental.preprocessing.RandomRotation(angle)(image)
        
        # Random zoom
        zoom = tf.random.uniform([], 1-self.config.ZOOM_RANGE, 1+self.config.ZOOM_RANGE)
        image = tf.keras.layers.experimental.preprocessing.RandomZoom(zoom)(image)
        
        # Ensure values are in [0, 1]
        image = tf.clip_by_value(image, 0, 1)
        
        return image
    
    def _create_dataset(self, images: List[str], labels: List[int], is_training: bool) -> tf.data.Dataset:
        """Create a tensorflow dataset"""
        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        
        # Map preprocessing
        ds = ds.map(lambda x, y: (self._preprocess_image(x), y),
                   num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            # Shuffle and augment training data
            ds = ds.shuffle(buffer_size=len(images))
            ds = ds.map(lambda x, y: (self._augment(x), y),
                       num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        ds = ds.batch(self.config.BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds 