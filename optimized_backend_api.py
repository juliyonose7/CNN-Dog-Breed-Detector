"""
Optimized Backend API for Dog Classification
=============================================

Production-ready FastAPI server for dog detection and breed classification.
Uses trained binary (dog/not-dog) and breed classification models for
real-time image analysis.

Features:
    - Binary dog detection with 95.43% accuracy
    - Multi-breed classification with 91.99% accuracy
    - Top-5 breed predictions with confidence scores
    - Automatic GPU acceleration when available
    - Full CORS support for web frontend integration

Endpoints:
    GET  /          - Service information and status
    GET  /health    - System health and model status
    POST /classify  - Full classification (detection + breed)
    POST /detect    - Binary dog detection only

Usage:
    python optimized_backend_api.py
    # Server runs on http://localhost:8001

Author: Dog Classification Team
Version: 1.0.0
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

class OptimizedDogClassifier:
    """
    Optimized classifier combining binary detection and breed classification.
    
    Loads pre-trained models and provides methods for single-image inference
    with automatic preprocessing and GPU acceleration.
    
    Attributes:
        device (str): Computation device ('cuda' or 'cpu').
        binary_model: ResNet-18 model for dog detection.
        breed_model: ResNet-18 model for breed classification.
        transform: Image preprocessing transformations.
    """
    
    def __init__(self):
        """
        Initialize the classifier and load trained models.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Paths to best trained models
        self.binary_model_path = "enhanced_binary_models/best_model_epoch_1_acc_0.9543.pth"
        self.breed_model_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
        
        # Load models
        self.binary_model = self.load_binary_model()
        self.breed_model = self.load_breed_model()
        
        # Image preprocessing transformations (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Optimized classifier ready!")
    
    def load_binary_model(self):
        """
        Load the binary classification model (dog/not-dog).
        
        Returns:
            nn.Module: Loaded and configured ResNet-18 model, or None if loading fails.
        """
        try:
            if not Path(self.binary_model_path).exists():
                print(f"Binary model not found: {self.binary_model_path}")
                return None
            
            print(f"Loading binary model: {self.binary_model_path}")
            checkpoint = torch.load(self.binary_model_path, map_location=self.device)
            
            # Create ResNet-18 model for binary classification
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: dog/not-dog
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"Binary model loaded successfully (Accuracy: 95.43%)")
            return model
            
        except Exception as e:
            print(f"Error loading binary model: {e}")
            return None
    
    def load_breed_model(self):
        """
        Load the breed classification model.
        
        Returns:
            nn.Module: Loaded and configured ResNet-18 model, or None if loading fails.
        """
        try:
            if not Path(self.breed_model_path).exists():
                print(f"Warning: Breed model not found: {self.breed_model_path}")
                return None
            
            print(f"Loading breed model: {self.breed_model_path}")
            checkpoint = torch.load(self.breed_model_path, map_location=self.device)
            
            # Get number of classes from checkpoint
            num_classes = checkpoint.get('num_classes', 50)
            print(f"Number of breeds: {num_classes}")
            
            # Create ResNet-18 model for breed classification
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"Breed model loaded successfully (Accuracy: 91.99%)")
            return model
            
        except Exception as e:
            print(f"Error loading breed model: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Converts image to RGB, applies resizing, normalization, and
        adds batch dimension.
        
        Args:
            image (Image.Image): PIL Image to preprocess.
        
        Returns:
            torch.Tensor: Preprocessed tensor of shape (1, 3, 224, 224).
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict_binary(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Predict whether image contains a dog.
        
        Args:
            image (Image.Image): PIL Image to analyze.
        
        Returns:
            tuple: (is_dog, probability) where is_dog is boolean and
                   probability is the dog class confidence score.
        """
        if self.binary_model is None:
            return False, 0.0
        
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.binary_model(tensor)
            probs = F.softmax(outputs, dim=1)
            dog_prob = probs[0, 1].item()  # Probability of being dog (class 1)
            is_dog = dog_prob > 0.5
        
        return is_dog, dog_prob
    
    def predict_breed(self, image: Image.Image) -> Tuple[str, float, List[Dict]]:
        """
        Predict the breed of a dog in the image.
        
        Args:
            image (Image.Image): PIL Image containing a dog.
        
        Returns:
            tuple: (best_breed, best_confidence, top5_predictions) where
                   top5_predictions is a list of breed prediction dicts.
        """
        if self.breed_model is None:
            return "Breed model not available", 0.0, []
        
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.breed_model(tensor)
            probs = F.softmax(outputs, dim=1)
            
            # Get top-5 predictions
            top5_probs, top5_indices = torch.topk(probs, min(5, probs.size(1)))
            
            predictions = []
            for i in range(top5_probs.size(1)):
                class_idx = top5_indices[0, i].item()
                confidence = top5_probs[0, i].item()
                breed_name = f"Breed_{class_idx}"  # Placeholder - replace with actual breed mapping
                
                predictions.append({
                    'breed': breed_name,
                    'confidence': confidence,
                    'class_index': class_idx
                })
            
            best_breed = predictions[0]['breed']
            best_confidence = predictions[0]['confidence']
            
            return best_breed, best_confidence, predictions
    
    def classify(self, image: Image.Image) -> Dict:
        """
        Perform complete classification (detection + breed identification).
        
        First determines if the image contains a dog, then if positive,
        predicts the breed with top-5 confidence scores.
        
        Args:
            image (Image.Image): PIL Image to classify.
        
        Returns:
            dict: Complete classification results including detection status,
                  breed predictions, and processing metadata.
        """
        start_time = time.time()
        
        # Step 1: Is it a dog?
        is_dog, dog_confidence = self.predict_binary(image)
        
        result = {
            'is_dog': is_dog,
            'dog_confidence': round(dog_confidence, 4),
            'prediction': 'dog' if is_dog else 'not_dog',
            'processing_time_ms': 0,
            'breed_info': None
        }
        
        # Step 2: If dog detected, classify breed
        if is_dog and self.breed_model is not None:
            breed, breed_confidence, top5_breeds = self.predict_breed(image)
            result['breed_info'] = {
                'primary_breed': breed,
                'breed_confidence': round(breed_confidence, 4),
                'top5_breeds': [
                    {
                        'breed': pred['breed'],
                        'confidence': round(pred['confidence'], 4),
                        'class_index': pred['class_index']
                    }
                    for pred in top5_breeds
                ]
            }
        elif is_dog:
            result['breed_info'] = {
                'message': 'Breed model not available',
                'status': 'unavailable'
            }
        
        # Calculate processing time
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return result

# Create FastAPI application
app = FastAPI(
    title="Dog Classifier API - Optimized",
    description="API for dog detection and breed classification using trained models",
    version="1.0.0"
)

# Configure CORS to allow connections from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize the classifier when the API starts.
    
    Loads the trained models and prepares the system for inference.
    
    Raises:
        HTTPException: If classifier initialization fails.
    """
    global classifier
    try:
        print("Starting dog classification API...")
        classifier = OptimizedDogClassifier()
        print("API ready to receive requests!")
    except Exception as e:
        print(f"Error initializing API: {e}")
        raise HTTPException(500, f"Error initializing API: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint with service information.
    
    Returns:
        dict: Service metadata, available endpoints, and model performance stats.
    """
    return {
        "service": "Dog Classifier API - Optimized",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/classify": "Full classification (detection + breed)",
            "/detect": "Dog detection only",
            "/health": "System status",
            "/docs": "Interactive documentation"
        },
        "models": {
            "binary": "95.43% accuracy",
            "breed": "91.99% accuracy"
        }
    }


@app.get("/health")
async def health_check():
    """
    Check system health and model status.
    
    Returns:
        dict: Health status including model availability and device info.
    """
    global classifier
    
    return {
        "status": "healthy" if classifier is not None else "error",
        "binary_model_loaded": classifier is not None and classifier.binary_model is not None,
        "breed_model_loaded": classifier is not None and classifier.breed_model is not None,
        "device": classifier.device if classifier else "unknown",
        "timestamp": time.time()
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image for dog detection and breed identification.
    
    Performs two-stage classification:
    1. Binary detection (dog vs not-dog)
    2. Breed classification if dog detected
    
    Args:
        file (UploadFile): Image file to classify.
    
    Returns:
        dict: Complete classification results with confidence scores.
    
    Raises:
        HTTPException: 400 if file is not an image, 500 on processing error.
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Classifier not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Classify
        result = classifier.classify(image)
        
        # Add metadata
        result['filename'] = file.filename
        result['file_size_kb'] = len(image_data) // 1024
        result['image_size'] = image.size
        result['timestamp'] = time.time()
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")

@app.post("/detect")
async def detect_dog(file: UploadFile = File(...)):
    """
    Binary dog detection endpoint.
    
    Determines whether an image contains a dog without breed classification.
    Faster than full classification for applications only needing detection.
    
    Args:
        file (UploadFile): Image file to analyze.
    
    Returns:
        dict: Detection result with confidence score.
    
    Raises:
        HTTPException: 400 if file is not an image, 500 on processing error.
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Classifier not initialized")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        start_time = time.time()
        is_dog, confidence = classifier.predict_binary(image)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            'is_dog': is_dog,
            'confidence': round(confidence, 4),
            'prediction': 'dog' if is_dog else 'not_dog',
            'processing_time_ms': processing_time,
            'filename': file.filename,
            'timestamp': time.time()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

if __name__ == "__main__":
    print("Starting dog classification API...")
    print("Available endpoints:")
    print("   http://localhost:8001/classify - Full classification")
    print("   http://localhost:8001/detect - Detection only")
    print("   http://localhost:8001/health - System status")
    print("   http://localhost:8001/docs - Interactive API documentation")
    print("=" * 60)
    
    uvicorn.run(
        "optimized_backend_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )