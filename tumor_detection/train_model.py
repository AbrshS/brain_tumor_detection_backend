import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def create_dataset(data_dir):
    images = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        print(f"Loading {class_name} images...")
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                try:
                    img_path = os.path.join(class_dir, img_name)
                    # Convert to grayscale and maintain single channel
                    img = Image.open(img_path).convert('L')
                    img = img.resize(IMG_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=-1)
                    img_array = np.repeat(img_array, 3, axis=-1)
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def build_model(num_classes):
    # Use a smaller model (B0 instead of B7) to prevent overfitting
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)  # Add dropout to prevent overfitting
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def train_model():
    # Load and prepare data
    print("Loading training data...")
    data_dir = "data/Training"  # Adjust path as needed
    X, y = create_dataset(data_dir)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(range(len(CLASS_NAMES)), class_weights))
    
    # Build and compile model
    print("Building model...")
    model = build_model(len(CLASS_NAMES))
    
    # Use a lower learning rate
    optimizer = Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
    
    # Train model
    print("Training model...")
    history = model.fit(
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(1000)
            .map(lambda x, y: (data_augmentation(x, training=True), y))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Save training history
    np.save('models/training_history.npy', history.history)
    
    return model, history

if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    # Enable mixed precision training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Train model
    model, history = train_model() 