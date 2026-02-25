"""
Centralized Configuration Module for Dog Breed Classification System.
This module contains all configurable constants and parameters for the project.
Modify these settings to adjust system behavior without changing source code.
"""

# =============================================================================
# PROJECT METADATA
# =============================================================================
PROJECT_NAME = "Dog Classification API"  # Descriptive name for the project
VERSION = "1.0.0"                        # Current system version

# =============================================================================
# FILE SYSTEM PATHS
# All paths are relative to the project root directory
# =============================================================================
DATASET_PATH = "./DATASETS"              # Directory containing original images
PROCESSED_DATA_PATH = "./processed_data" # Directory for preprocessed data
MODELS_PATH = "./models"                 # Directory for saved model checkpoints
OPTIMIZED_MODELS_PATH = "./optimized_models"  # Production-optimized models

# =============================================================================
# NEURAL NETWORK MODEL CONFIGURATION
# Defines the architecture and fundamental parameters
# =============================================================================
MODEL_CONFIG = {
    "model_name": "efficientnet_b3",  # Architecture: efficientnet_b3, resnet50, resnet101, densenet121
    "input_size": (224, 224),         # Input image resolution in pixels
    "num_classes": 1,                 # Number of initial classification classes
    "pretrained": True                # Use ImageNet pretrained weights
}

# =============================================================================
# TRAINING HYPERPARAMETERS
# Controls optimization strategies and learning dynamics
# =============================================================================
TRAINING_CONFIG = {
    "batch_size": 32,                 # Number of samples processed simultaneously per batch
    "num_epochs": 30,                 # Total number of training epochs
    "learning_rate": 1e-3,            # Initial optimizer learning rate
    "weight_decay": 1e-4,             # L2 regularization to prevent overfitting
    "freeze_epochs": 5,               # Epochs with frozen backbone at start
    "balance_strategy": "undersample" # Class balancing: undersample, oversample, none
}

# =============================================================================
# AMD ROCM/DIRECTML HARDWARE OPTIMIZATION
# Adjusts parameters for optimal performance on AMD GPUs
# =============================================================================
ROCM_CONFIG = {
    "device": "cuda",              # Device: cuda for ROCm/DirectML, cpu for fallback
    "mixed_precision": True,       # Mixed precision training for faster throughput
    "benchmark": True,             # Optimize cuDNN for best performance
    "deterministic": False         # False allows faster speed, sacrificing reproducibility
}

# =============================================================================
# DATA AUGMENTATION CONFIGURATION
# Defines random transformations applied during training
# =============================================================================
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,        # Probability of horizontal flip
    "rotation_limit": 15,          # Maximum rotation in degrees
    "brightness_contrast": 0.2,    # Brightness and contrast variation
    "gaussian_noise": 0.3,         # Gaussian noise for robustness
    "cutout": 0.2,                 # Cutout regularization probability
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet normalization mean
    "normalize_std": [0.229, 0.224, 0.225]    # ImageNet normalization std dev
}

# =============================================================================
# REST API SERVER CONFIGURATION
# Defines network parameters and security limits
# =============================================================================
API_CONFIG = {
    "host": "0.0.0.0",                      # Accept connections from any IP
    "port": 8000,                           # Server listening port
    "workers": 1,                           # Number of uvicorn workers
    "max_file_size": 10 * 1024 * 1024,     # Maximum file size: 10MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],  # Allowed image formats
    "max_batch_size": 10                   # Maximum images per batch request
}

# =============================================================================
# LOGGING CONFIGURATION
# Controls event and error recording for debugging
# =============================================================================
LOGGING_CONFIG = {
    "level": "INFO",                        # Minimum logging level
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log message format
    "log_file": "./logs/app.log"           # Log output file path
}