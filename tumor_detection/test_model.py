import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_PATH = 'models/best_effnetb7_model.h5'
TEST_DATA_PATH = 'test_data'  # Create this directory and put test images in respective class folders
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_and_preprocess_image(image_path):
    """Load and preprocess image exactly as in training"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.repeat(img_array, 3, axis=-1)  # Convert to 3 channels
    return np.expand_dims(img_array, axis=0)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def test_model():
    # Load model
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    
    # Prepare data
    X_test = []
    y_true = []
    file_paths = []
    
    print("\nProcessing test images...")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_DATA_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img_array = load_and_preprocess_image(img_path)
                    X_test.append(img_array)
                    y_true.append(class_idx)
                    file_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    if not X_test:
        print("No test images found!")
        return
    
    X_test = np.vstack(X_test)
    y_true = np.array(y_true)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 50)
    for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
        confidence = predictions[i][pred_idx]
        print(f"\nImage: {os.path.basename(file_paths[i])}")
        print(f"True class: {CLASS_NAMES[true_idx]}")
        print(f"Predicted: {CLASS_NAMES[pred_idx]} (confidence: {confidence:.2%})")
        print(f"All probabilities:")
        for class_name, prob in zip(CLASS_NAMES, predictions[i]):
            print(f"  {class_name}: {prob:.2%}")
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    # Enable mixed precision as used in training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    test_model() 