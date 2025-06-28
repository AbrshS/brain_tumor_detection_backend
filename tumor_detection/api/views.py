import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Add, UpSampling2D, Concatenate
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
SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, 'ResUNet-bestSegModel-weights.h5')    

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Model directory: {MODEL_DIR}")
logger.info(f"Classification model path: {CLASSIFICATION_MODEL_PATH}")
logger.info(f"Segmentation model path: {SEGMENTATION_MODEL_PATH}")

def build_unet_architecture():
    """Recreate the exact ResUNet architecture used in training"""
    def resblock(X, f):
        X_copy = X  # Copy of input
        
        # Main path
        X = Conv2D(f, kernel_size=(1,1), kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        
        X = Conv2D(f, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        
        # Shortcut path
        X_copy = Conv2D(f, kernel_size=(1,1), kernel_initializer='he_normal')(X_copy)
        X_copy = BatchNormalization()(X_copy)
        
        # Adding the output from main path and short path together
        X = Add()([X, X_copy])
        X = Activation('relu')(X)
        return X

    def upsample_concat(x, skip):
        X = UpSampling2D((2,2))(x)
        merge = Concatenate()([X, skip])
        return merge

    # Input
    input_shape = (256, 256, 3)
    X_input = Input(input_shape)

    # Stage 1
    conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(X_input)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPooling2D((2,2))(conv_1)

    # Stage 2
    conv_2 = resblock(pool_1, 32)
    pool_2 = MaxPooling2D((2,2))(conv_2)

    # Stage 3
    conv_3 = resblock(pool_2, 64)
    pool_3 = MaxPooling2D((2,2))(conv_3)

    # Stage 4
    conv_4 = resblock(pool_3, 128)
    pool_4 = MaxPooling2D((2,2))(conv_4)

    # Stage 5 (bottle neck)
    conv_5 = resblock(pool_4, 256)

    # Upsample Stage 1
    up_1 = upsample_concat(conv_5, conv_4)
    up_1 = resblock(up_1, 128)

    # Upsample Stage 2
    up_2 = upsample_concat(up_1, conv_3)
    up_2 = resblock(up_2, 64)

    # Upsample Stage 3
    up_3 = upsample_concat(up_2, conv_2)
    up_3 = resblock(up_3, 32)

    # Upsample Stage 4
    up_4 = upsample_concat(up_3, conv_1)
    up_4 = resblock(up_4, 16)

    # Final output
    outputs = Conv2D(1, (1,1), kernel_initializer='he_normal', padding='same', activation='sigmoid')(up_4)
    
    model = tf.keras.Model(inputs=X_input, outputs=outputs)
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
        
        # First try: Load as complete model
        try:
            logger.info("Attempting to load as complete model...")
            model = load_model(model_path, compile=False)
            logger.info("Successfully loaded segmentation model as complete model!")
            return model
        except Exception as e:
            logger.warning(f"Failed to load as complete model: {str(e)}")
            logger.warning("Attempting to load as weights file...")
            
            # Second try: Load architecture and weights separately
            try:
                logger.info("Building fresh architecture...")
                model = build_unet_architecture()
                logger.info("Loading weights...")
                model.load_weights(model_path)
                logger.info("Successfully loaded segmentation model with weights!")
                return model
            except Exception as e2:
                logger.error(f"Failed to load weights: {str(e2)}")
                logger.error(traceback.format_exc())
                raise
            
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
            img_array_classification, img_array_segmentation, original_img = self._preprocess_image(img_file)
            
            # Classify tumor
            class_name, confidence = self._classify_tumor(img_array_classification)
            
            # Analyze tumor if present
            analysis = None
            if class_name != 'notumor':
                analysis = self._analyze_tumor(img_array_segmentation, original_img)
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
            
            # For classification model
            img_classification = img.resize((224, 224))
            img_array_classification = np.array(img_classification, dtype=np.float32)
            
            # For segmentation model - match training preprocessing exactly
            img_segmentation = img.resize((256, 256))
            img_array_segmentation = np.array(img_segmentation, dtype=np.float64)
            
            # Standardize like in training
            img_array_segmentation -= img_array_segmentation.mean()
            img_array_segmentation /= img_array_segmentation.std()
            
            # Add batch dimension for model input
            img_array_segmentation = np.expand_dims(img_array_segmentation, axis=0)
            
            logger.info(f"Classification input shape: {img_array_classification.shape}")
            logger.info(f"Segmentation input shape: {img_array_segmentation.shape}")
            logger.info(f"Segmentation input range: {img_array_segmentation.min():.2f} to {img_array_segmentation.max():.2f}")
            
            return (np.expand_dims(img_array_classification, axis=0), 
                   img_array_segmentation,
                   original_img)
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

    def _analyze_tumor(self, img_array_segmentation, original_img):
        """Complete tumor segmentation and analysis"""
        try:
            # Generate mask using preprocessed input
            try:
                logger.info(f"Running segmentation prediction on input shape: {img_array_segmentation.shape}")
                mask_pred = segmentation_model.predict(img_array_segmentation, verbose=0)
                logger.info(f"Mask prediction shape: {mask_pred.shape}")
                logger.info(f"Mask prediction range: {mask_pred.min():.2f} to {mask_pred.max():.2f}")
                
                # Threshold the prediction
                mask = (mask_pred[0,...,0] > 0.5).astype(np.uint8)
            except Exception as e:
                logger.error(f"Segmentation prediction failed: {str(e)}")
                logger.error(traceback.format_exc())
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