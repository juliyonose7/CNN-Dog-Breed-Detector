"""
Hierarchical Dog Classification API Module
==========================================

This module implements a FastAPI-based REST API for hierarchical dog classification.
The system uses a two-stage approach:
1. Binary classification: Detect if the image contains a dog
2. Breed classification: If a dog is detected, identify among 50+ breeds

Features:
    - FastAPI with automatic OpenAPI documentation
    - CORS support for cross-origin requests
    - Multiple endpoints for flexible usage:
        - /classify: Full hierarchical classification
        - /detect-dog: Binary dog detection only
        - /classify-breed: Breed classification only
    - Top-5 breed predictions with confidence scores
    - Health check and system status endpoints

Endpoints:
    GET  /          - API information and available endpoints
    GET  /health    - System health check
    POST /classify  - Full hierarchical classification (dog + breed)
    POST /detect-dog - Dog detection only
    POST /classify-breed - Breed classification only (assumes dog)
    GET  /breeds    - List all available breeds

Author: AI System
Date: 2024
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
import io
import base64

import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

class HierarchicalDogClassifier:
    """
    Hierarchical dog classifier combining binary detection and breed classification.
    
    This classifier implements a two-stage inference pipeline:
    1. Binary model determines if an image contains a dog
    2. If a dog is detected, breed model identifies the specific breed
    
    Args:
        binary_model_path (str): Path to the binary classification model weights.
        breed_model_path (str, optional): Path to the breed classification model weights.
    
    Attributes:
        device (str): Computation device ('cuda' or 'cpu').
        binary_model: Loaded binary classification model.
        breed_model: Loaded breed classification model (or None if unavailable).
        breed_names (dict): Mapping from class index to breed name.
        transform: Image preprocessing pipeline.
    """
    
    def __init__(self, binary_model_path: str, breed_model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load binary model (dog vs not-dog)
        print(" Loading binary model...")
        self.binary_model = self.load_binary_model(binary_model_path)
        
        # Load breed classification model if available
        self.breed_model = None
        self.breed_names = {}
        
        if breed_model_path and Path(breed_model_path).exists():
            print(" Loading breed model...")
            self.breed_model = self.load_breed_model(breed_model_path)
        else:
            print("  Breed model not available yet")
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f" Models loaded on device: {self.device}")
    
    def load_binary_model(self, model_path: str):
        """
        Load the binary classification model.
        
        Args:
            model_path (str): Path to the model checkpoint file.
        
        Returns:
            nn.Module: Loaded binary model in evaluation mode.
        
        Raises:
            Exception: If model loading fails.
        """
        try:
            # Recreate binary model architecture
            from quick_train import DogClassificationModel
            
            model = DogClassificationModel()
            
            # Load weights from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            print(" Binary model loaded successfully")
            return model
            
        except Exception as e:
            print(f" Error loading binary model: {e}")
            raise
    
    def load_breed_model(self, model_path: str):
        """
        Load the breed classification model.
        
        Args:
            model_path (str): Path to the breed model checkpoint file.
        
        Returns:
            nn.Module: Loaded breed model in evaluation mode, or None on failure.
        """
        try:
            # Load checkpoint containing model config
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            model_config = checkpoint.get('model_config', {})
            num_classes = model_config.get('num_classes', 50)
            model_name = model_config.get('model_name', 'efficientnet_b3')
            self.breed_names = model_config.get('breed_names', {})
            
            # Recreate model architecture
            from breed_trainer import AdvancedBreedClassifier
            
            model = AdvancedBreedClassifier(
                num_classes=num_classes,
                model_name=model_name,
                pretrained=False
            )
            
            # Load trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f" Breed model loaded: {num_classes} classes")
            return model
            
        except Exception as e:
            print(f" Error loading breed model: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Converts the image to RGB, applies transforms, and prepares
        the tensor for batch inference.
        
        Args:
            image (PIL.Image): Input image to preprocess.
        
        Returns:
            torch.Tensor: Preprocessed image tensor on the target device.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def predict_binary(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Predict whether the image contains a dog (Stage 1).
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            Tuple[bool, float]: (is_dog, confidence) where is_dog is True if
                                a dog is detected and confidence is the probability.
        """
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.binary_model(tensor)
            
            # Handle binary model with sigmoid output
            if output.shape[1] == 1:
                prob = torch.sigmoid(output).item()
                is_dog = prob > 0.5
            else:
                # Handle model with 2-class output
                probs = F.softmax(output, dim=1)
                prob = probs[0, 1].item()  # Probability of being a dog
                is_dog = prob > 0.5
        
        return is_dog, prob
    
    def predict_breed(self, image: Image.Image) -> Tuple[str, float, List[Dict]]:
        """
        Predict the dog breed (Stage 2).
        
        Args:
            image (PIL.Image): Input image (should contain a dog).
        
        Returns:
            Tuple[str, float, List[Dict]]: (best_breed, confidence, top5_predictions)
                where top5_predictions is a list of dicts with 'breed', 'confidence',
                and 'class_index' keys.
        """
        if self.breed_model is None:
            return "Breed model not available", 0.0, []
        
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.breed_model(tensor)
            probs = F.softmax(output, dim=1)
            
            # top-5 predictions
            top5_probs, top5_indices = torch.topk(probs, min(5, probs.size(1)))
            
            predictions = []
            for i in range(top5_probs.size(1)):
                class_idx = top5_indices[0, i].item()
                confidence = top5_probs[0, i].item()
                breed_name = self.breed_names.get(class_idx, f"Raza_{class_idx}")
                
                predictions.append({
                    'breed': breed_name,
                    'confidence': confidence,
                    'class_index': class_idx
                })
            
            # Best prediction
            best_breed = predictions[0]['breed']
            best_confidence = predictions[0]['confidence']
            
            return best_breed, best_confidence, predictions
    
    def classify_hierarchical(self, image: Image.Image) -> Dict:
        """
        Perform complete hierarchical classification pipeline.
        
        Executes both binary detection and breed classification in sequence,
        only proceeding to breed classification if a dog is detected.
        
        Args:
            image (PIL.Image): Input image to classify.
        
        Returns:
            dict: Classification results containing:
                - is_dog (bool): Whether a dog was detected
                - dog_confidence (float): Binary model confidence
                - processing_time_ms (int): Total processing time
                - breed_info (dict or None): Breed classification results
        """
        start_time = time.time()
        
        # Stage 1: Is it a dog?
        is_dog, dog_confidence = self.predict_binary(image)
        
        result = {
            'is_dog': is_dog,
            'dog_confidence': dog_confidence,
            'processing_time_ms': 0,
            'breed_info': None
        }
        
        # Stage 2: Breed classification if dog detected
        if is_dog and self.breed_model is not None:
            breed, breed_confidence, top5_breeds = self.predict_breed(image)
            
            result['breed_info'] = {
                'primary_breed': breed,
                'breed_confidence': breed_confidence,
                'top5_breeds': top5_breeds
            }
        elif is_dog and self.breed_model is None:
            result['breed_info'] = {
                'message': 'Breed model in training...',
                'status': 'training'
            }
        
        # Time of processing
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        
        return result

# Initialize FastAPI application
app = FastAPI(
    title="Hierarchical Dog Classification API",
    description="Hierarchical system: detects dogs and classifies breeds",
    version="2.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize the classifier on application startup.
    
    Loads both binary and breed models and prepares the system for inference.
    
    Raises:
        HTTPException: If binary model is not found or initialization fails.
    """
    global classifier
    
    # Model file paths
    binary_model_path = "best_model.pth"
    breed_model_path = "breed_models/best_breed_model.pth"
    
    # Verify binary model exists
    if not Path(binary_model_path).exists():
        print(f" Binary model not found: {binary_model_path}")
        raise HTTPException(500, "Binary model not available")
    
    try:
        classifier = HierarchicalDogClassifier(
            binary_model_path=binary_model_path,
            breed_model_path=breed_model_path
        )
        print(" Hierarchical API ready!")
    except Exception as e:
        print(f" Error initializing classifier: {e}")
        raise HTTPException(500, f"Error initializing models: {str(e)}")

@app.get("/")
async def root():
    """
    Get API information and available endpoints.
    
    Returns:
        dict: API service information including version, status, and endpoint list.
    """
    return {
        "service": "Hierarchical Dog Classification API",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "Binary dog/not-dog detection",
            "Classification of 50 dog breeds",
            "Optimized hierarchical system",
            "Top-5 breed predictions"
        ],
        "endpoints": {
            "/classify": "Full hierarchical classification",
            "/detect-dog": "Dog detection only",
            "/classify-breed": "Breed classification only (requires dog image)",
            "/health": "System status"
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
    
    status = {
        "status": "healthy",
        "binary_model": classifier is not None,
        "breed_model": classifier is not None and classifier.breed_model is not None,
        "device": classifier.device if classifier else "unknown",
        "breed_classes": len(classifier.breed_names) if classifier and classifier.breed_names else 0
    }
    
    return status

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Perform complete hierarchical classification.
    
    First detects if the image contains a dog, then if positive,
    classifies the breed with top-5 predictions.
    
    Args:
        file (UploadFile): Image file to classify.
    
    Returns:
        dict: Classification results with dog detection and breed info.
    
    Raises:
        HTTPException: If classifier not initialized or processing fails.
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Classifier not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Perform classification
        result = classifier.classify_hierarchical(image)
        
        # Add file metadata
        result['filename'] = file.filename
        result['file_size_kb'] = len(image_data) // 1024
        result['image_dimensions'] = image.size
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")

@app.post("/detect-dog")
async def detect_dog_only(file: UploadFile = File(...)):
    """
    Detect only whether the image contains a dog (Stage 1 only).
    
    Args:
        file (UploadFile): Image file to analyze.
    
    Returns:
        dict: Dog detection result with confidence score.
    
    Raises:
        HTTPException: If classifier not initialized or processing fails.
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
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'is_dog': is_dog,
            'confidence': confidence,
            'class': 'dog' if is_dog else 'not_dog',
            'processing_time_ms': processing_time,
            'filename': file.filename
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")

@app.post("/classify-breed")
async def classify_breed_only(file: UploadFile = File(...)):
    """
    Classify breed only (assumes image contains a dog).
    
    Skips binary detection and proceeds directly to breed classification.
    Use when you know the image contains a dog.
    
    Args:
        file (UploadFile): Image file containing a dog.
    
    Returns:
        dict: Breed classification with top-5 predictions.
    
    Raises:
        HTTPException: If breed model unavailable or processing fails.
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Classifier not initialized")
    
    if classifier.breed_model is None:
        raise HTTPException(503, "Breed model not available (training...)")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        start_time = time.time()
        breed, confidence, top5_breeds = classifier.predict_breed(image)
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'primary_breed': breed,
            'breed_confidence': confidence,
            'top5_breeds': top5_breeds,
            'processing_time_ms': processing_time,
            'filename': file.filename
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error classifying breed: {str(e)}")

@app.get("/breeds")
async def list_breeds():
    """
    List all available dog breeds.
    
    Returns:
        dict: Status and list of breed names with display names.
    """
    global classifier
    
    if classifier is None or classifier.breed_model is None:
        return {
            'status': 'not_available',
            'message': 'Breed model not available',
            'breeds': []
        }
    
    breeds = []
    for idx, name in classifier.breed_names.items():
        breeds.append({
            'index': idx,
            'name': name,
            'display_name': name.replace('_', ' ').title()
        })
    
    return {
        'status': 'available',
        'total_breeds': len(breeds),
        'breeds': sorted(breeds, key=lambda x: x['name'])
    }

if __name__ == "__main__":
    print("Starting hierarchical API...")
    print("Available endpoints:")
    print("   http://localhost:8001/classify - Full classification")
    print("   http://localhost:8001/detect-dog - Dog detection only")
    print("   http://localhost:8001/classify-breed - Breed classification only")
    print("   http://localhost:8001/docs - API documentation")
    
    uvicorn.run(
        "hierarchical_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )