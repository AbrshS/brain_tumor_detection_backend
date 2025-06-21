import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.conf import settings
from PIL import Image
import cv2
import logging
import sys
import traceback
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, models
from rest_framework import generics, permissions, status
from rest_framework.authtoken.models import Token
from .serializers import UserRegistrationSerializer, LoginSerializer
from django.contrib.auth import authenticate


# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model paths (using relative paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, 'effnetb7_perfect_model.h5')
SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, 'brain_tumor_segmentation.h5')

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Model directory: {MODEL_DIR}")
logger.info(f"Classification model path: {CLASSIFICATION_MODEL_PATH}")
logger.info(f"Segmentation model path: {SEGMENTATION_MODEL_PATH}")

def build_unet_architecture():
    """Recreate the exact U-Net architecture used in training"""
    inputs = Input((256, 256, 1), dtype=tf.float32)
    
    # Downsample path
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)
    
    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)
    
    # Bottleneck
    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
    
    # Upsample path
    u4 = Conv2DTranspose(64, 2, strides=2, padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)
    
    u5 = Conv2DTranspose(32, 2, strides=2, padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(32, 3, activation='relu', padding='same')(c5)
    
    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_segmentation_model(model_path):
    """Dedicated loader for segmentation model"""
    try:
        logger.info(f"Attempting to load segmentation model from: {model_path}")
        logger.info(f"File exists: {os.path.exists(model_path)}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        file_size = os.path.getsize(model_path)
        logger.info(f"Segmentation model file size: {file_size / (1024*1024):.2f} MB")
        
        # Try loading with minimal configuration
        try:
            logger.info("Attempting to load segmentation model with minimal config...")
            model = load_model(model_path, compile=False)
            logger.info("Successfully loaded segmentation model!")
            return model
        except Exception as e:
            logger.error(f"Failed to load with minimal config: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback: Try loading with exact architecture
            logger.info("Attempting fallback with exact training architecture...")
            model = build_unet_architecture()
            model.load_weights(model_path)
            logger.info("Successfully loaded segmentation model with exact architecture!")
            return model
            
    except Exception as e:
        logger.critical(f"Failed to load segmentation model: {str(e)}")
        logger.critical(traceback.format_exc())
        return None

def load_classification_model(model_path):
    """Dedicated loader for classification model"""
    try:
        logger.info(f"Attempting to load classification model from: {model_path}")
        logger.info(f"File exists check: {os.path.exists(model_path)}")
        logger.info(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Try loading with custom objects
        custom_objects = {
            'FixedDropout': tf.keras.layers.Dropout,
            'Dropout': tf.keras.layers.Dropout,
            'Dense': tf.keras.layers.Dense,
            'BatchNormalization': tf.keras.layers.BatchNormalization
        }
        
        try:
            model = load_model(model_path, compile=False, custom_objects=custom_objects)
            logger.info("Successfully loaded classification model!")
            return model
        except Exception as inner_e:
            logger.error(f"Initial load attempt failed: {str(inner_e)}")
            logger.error(traceback.format_exc())
            
            # Try alternative loading method
            logger.info("Attempting alternative loading method...")
            base_model = EfficientNetB7(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
            base_model.trainable = False

            inputs = layers.Input(shape=(224, 224, 3))
            x = tf.keras.applications.efficientnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(4, activation='softmax')(x)
            model = models.Model(inputs, outputs)
            model.load_weights(model_path)
            logger.info("Successfully loaded model with alternative method!")
            return model
            
    except Exception as e:
        logger.critical(f"Failed to load classification model: {str(e)}")
        logger.critical(traceback.format_exc())
        logger.critical(f"Python version: {sys.version}")
        logger.critical(f"TensorFlow version: {tf.version}")
        return None


# Initialize models at startup
try:
    logger.info("=== Starting model initialization ===")
    
    logger.info("Loading classification model...")
    classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)
    
    logger.info("Loading segmentation model...")
    segmentation_model = load_segmentation_model(SEGMENTATION_MODEL_PATH)
    
    if classification_model is None:
        logger.critical("Classification model failed to load!")
    if segmentation_model is None:
        logger.critical("Segmentation model failed to load!")
        
    logger.info("=== Model initialization complete ===")
except Exception as e:
    logger.critical(f"Critical error during model initialization: {str(e)}")
    logger.critical(traceback.format_exc())
    classification_model = None
    segmentation_model = None

class TumorAnalysisAPI(APIView):
    parser_classes = [MultiPartParser]
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    PIXEL_TO_MM = 0.264583  # 1 pixel = 0.264583 mm (DICOM standard)

    def post(self, request, *args, **kwargs):
        """Endpoint for complete tumor analysis"""
        try:
            # Verify model availability
            if not classification_model:
                return Response({
                    "status": "error",
                    "message": "Classification model unavailable",
                    "solution": "Check server logs for loading errors"
                }, status=503)

            if not segmentation_model:
                return Response({
                    "status": "error",
                    "message": "Segmentation model unavailable",
                    "solution": "Check server logs for loading errors"
                }, status=503)

            # Validate input
            if 'file' not in request.FILES:
                return Response({
                    "status": "error",
                    "message": "No MRI file provided"
                }, status=400)

            # Process the image
            img_file = request.FILES['file']
            img_array, original_img = self._preprocess_image(img_file)
            
            # Classify tumor
            class_name, confidence = self._classify_tumor(img_array)
            
            # Analyze tumor if present
            analysis = None
            if class_name != 'notumor':
                analysis = self._analyze_tumor(original_img)
                if analysis:
                    analysis['visualization'] = self._generate_visualization(
                        original_img, 
                        analysis['mask']
                    )

            return Response({
                "status": "success",
                "prediction": class_name,
                "confidence": float(confidence),
                "analysis": analysis
            })

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "message": "Analysis failed",
                "error": str(e)
            }, status=500)

    def _preprocess_image(self, img_file):
        """Prepare image for both classification and segmentation"""
        try:
            # Load image in RGB mode to match training conditions
            img = Image.open(img_file).convert('RGB')
            original_img = np.array(img)
            
            # Resize to match model input size (224x224) while keeping RGB channels
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Keep original scaling (0-255) to match training conditions
            # Add batch dimension for model input
            return np.expand_dims(img_array, axis=0), original_img
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _classify_tumor(self, img_array):
        """Run classification prediction"""
        try:
            logger.info(f"Input shape: {img_array.shape}")
            logger.info(f"Input value range: {np.min(img_array)} to {np.max(img_array)}")
            
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            preds = classification_model.predict(img_array, verbose=0)[0]
            logger.info(f"Raw predictions: {preds}")
            logger.info(f"Prediction probabilities: {dict(zip(self.CLASS_NAMES, preds))}")
            
            class_idx = np.argmax(preds)
            return self.CLASS_NAMES[class_idx], float(preds[class_idx])
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _analyze_tumor(self, original_img):
        """Complete tumor segmentation and analysis"""
        try:
            # Convert to grayscale for segmentation model
            if len(original_img.shape) == 3:
                img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = original_img
                
            # Resize to 256x256 for segmentation model
            img = cv2.resize(img_gray, (256, 256))
            img_normalized = img.astype(np.float32) / 255.0
            img_normalized = np.expand_dims(img_normalized, axis=[0, -1])
            
            logger.info(f"Segmentation input shape: {img_normalized.shape}")
            
            # Generate mask
            try:
                mask_pred = segmentation_model.predict(img_normalized, verbose=0)
                logger.info(f"Mask prediction shape: {mask_pred.shape}")
                mask = (mask_pred[0,...,0] > 0.5).astype(np.uint8)
            except Exception as e:
                logger.error(f"Segmentation prediction failed: {str(e)}")
                return None
            
            # Post-processing
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate metrics
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.warning("No tumor regions found in segmentation mask")
                return None
                
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
            
            # Resize mask back to original dimensions for visualization
            mask_original_size = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
            
            return {
                "size": self._classify_size(area/(256*256)),
                "shape": "regular" if circularity > 0.7 else "irregular",
                "area_mm2": round(area * (self.PIXEL_TO_MM**2), 2),
                "perimeter_mm": round(perimeter * self.PIXEL_TO_MM, 2),
                "circularity": round(circularity, 3),
                "mask": mask_original_size
            }
        except Exception as e:
            logger.error(f"Tumor analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _classify_size(self, ratio):
        """Clinical size classification"""
        if ratio < 0.1: return "small"
        elif ratio < 0.3: return "medium"
        return "large"

    def _generate_visualization(self, original_img, mask):
        """Generate and save tumor overlay"""
        try:
            # Ensure mask and image have same dimensions
            if mask.shape[:2] != original_img.shape[:2]:
                logger.error(f"Mask shape {mask.shape} doesn't match image shape {original_img.shape}")
                return None
            
            # Create RGB version of original image if it's grayscale
            if len(original_img.shape) == 2:
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            else:
                original_rgb = original_img.copy()
            
            # Create overlay
            overlay = original_rgb.copy()
            overlay[mask == 1] = [255, 0, 0]  # Red highlight for tumor
            
            # Blend original and overlay
            alpha = 0.6
            result = cv2.addWeighted(original_rgb, 1-alpha, overlay, alpha, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            # Ensure directory exists
            vis_dir = os.path.join(settings.MEDIA_ROOT, 'tumor_visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save with unique filename
            filename = f"result_{os.urandom(4).hex()}.jpg"
            save_path = os.path.join(vis_dir, filename)
            
            # Save image
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization to: {save_path}")
            
            # Return URL path
            return os.path.join(settings.MEDIA_URL, 'tumor_visualizations', filename)
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

class RegisterView(generics.CreateAPIView):
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                "user": serializer.data,
                "token": token.key
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = LoginSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                "token": token.key,
                "user": {
                    "email": user.email,
                    "full_name": user.full_name,
                    "hospital_name": user.hospital_name,
                    "hospital_department": user.hospital_department
                }
            })
        return Response(serializer.errors, status=status.HTTP_401_UNAUTHORIZED)