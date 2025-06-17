# Brain Tumor Detection Training Pipeline

This is a production-ready training pipeline for brain tumor detection using EfficientNetB7.

## Features

- EfficientNetB7 architecture with custom top layers
- Extensive data augmentation
- Class weight balancing
- Mixed precision training
- XLA optimization
- Comprehensive logging with TensorBoard and W&B
- Model checkpointing and early stopping
- Learning rate scheduling
- Proper train/validation/test splits

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment:
- Create a Weights & Biases account and login:
```bash
wandb login
```
- Update paths in `config.py` if needed

3. Prepare your data:
- Organize your data in the following structure:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Training

To start training:

```bash
python -m tumor_detection.training.train
```

## Model Architecture

- Base: EfficientNetB7 (pretrained on ImageNet)
- Custom top layers:
  * Global Average Pooling
  * Dense (1024) + ReLU + Dropout (0.5)
  * Dense (512) + ReLU + Dropout (0.3)
  * Dense (4) + Softmax

## Data Preprocessing

- Resize to 224x224
- Convert to grayscale and back to RGB
- Normalize to [0, 1]

## Data Augmentation

- Random horizontal flips
- Random rotation (±20°)
- Random zoom (±20%)
- Random brightness/contrast adjustments
- Random shifts

## Training Strategy

- Fine-tuning last 20 layers of EfficientNetB7
- Class-weighted loss to handle imbalance
- Early stopping with patience of 10
- Learning rate reduction on plateau
- Mixed precision training for better performance
- Automatic model checkpointing

## Monitoring

Training progress can be monitored through:
- Weights & Biases dashboard
- TensorBoard logs
- CSV logs
- Console output with detailed logging

## Model Outputs

The training pipeline saves:
- Best model weights
- Checkpoints at regular intervals
- Training logs
- Evaluation metrics

## Performance Optimization

- Mixed precision training (float16)
- XLA compilation
- Automatic GPU memory growth
- TensorFlow data pipeline optimization
- Proper batch size and prefetching 