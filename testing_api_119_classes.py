#!/usr/bin/env python3
"""
Dog Breed Classification API Server (119 Classes).

This module implements a FastAPI-based REST API server for dog breed classification.
It serves a ResNet50 model trained on a balanced dataset of 119 dog breed classes
with adaptive thresholds for improved precision.

Key Features:
    - ResNet50 model trained with K-fold cross validation
    - 119 balanced breed classes from ImageNet
    - Adaptive confidence thresholds per breed
    - REST API with endpoints for prediction and model info
    - Support for JPG, PNG, and WEBP image formats
    - JSON responses with top-k predictions and confidence levels

Endpoints:
    GET /: API information
    GET /health: Server health check
    GET /model-info: Model details and configuration
    GET /breeds: List all classifiable breeds
    POST /predict: Breed prediction from uploaded image

Author: AI System
Date: 2024
"""

# Standard library imports for system operations and file handling
import os          # Operating system operations
import time        # Processing time measurement
import json        # JSON data handling
from pathlib import Path  # Modern path handling
from typing import Dict, List, Optional  # Type annotations

# PyTorch imports for deep learning
import torch                           # Main framework
import torch.nn as nn                  # Neural network modules
import torchvision.transforms as transforms  # Image transformations
import torchvision.models as models    # Pre-trained models

# FastAPI imports for web API
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # CORS support
from fastapi.responses import JSONResponse          # JSON responses

# Image processing imports
from PIL import Image    # Python image library
import io               # Input/output operations
import numpy as np      # Numerical operations
import uvicorn          # ASGI server for FastAPI

# =============================================================================
# MAIN CONFIGURATION: Model and API Server Settings
# =============================================================================
# These constants control the behavior of breed classification

# Path to the best trained model from k-fold cross validation
MODEL_PATH = "best_model_fold_0.pth"  # Best model from k-fold cross validation

# Total number of classes the model can classify
NUM_CLASSES = 119  # Balanced breeds in the final dataset

# Minimum confidence threshold to consider a prediction valid
CONFIDENCE_THRESHOLD = 0.1  # Very low to capture all possibilities

# Number of top predictions to return in the response
TOP_K_PREDICTIONS = 5  # Sufficient to show alternatives to user

# Port where the API server runs
API_PORT = 8000  # Standard development port

# =============================================================================
# ADAPTIVE THRESHOLDS: Per-breed confidence thresholds for improved precision
# =============================================================================
# Each breed has its own threshold based on historical performance.
# Higher thresholds = stricter requirements for that specific breed.
ADAPTIVE_THRESHOLDS = {
    'Lhasa': 0.35,                # Lhasa Apso requires 35% minimum
    'cairn': 0.40,                # Cairn Terrier requires 40% minimum
    'Siberian_husky': 0.45,       # Siberian Husky requires 45% minimum
    'whippet': 0.45,              # Whippet requires 45% minimum
    'malamute': 0.50,             # Malamute requires 50% minimum
    'Australian_terrier': 0.50,   # Australian Terrier requires 50% minimum
    'Norfolk_terrier': 0.50,      # Norfolk Terrier requires 50% minimum
    'toy_terrier': 0.55,          # Toy Terrier requires 55% minimum
    'Italian_greyhound': 0.55,    # Italian Greyhound requires 55% minimum
    'Lakeland_terrier': 0.55,     # Lakeland Terrier requires 55% minimum
    'bluetick': 0.55,             # Bluetick Coonhound requires 55% minimum
    'Border_terrier': 0.55,      
}

# Default threshold for definitive classification
DEFAULT_CLASSIFICATION_THRESHOLD = 0.60

# Image transformations (must match training transformations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Complete list of 119 balanced breed classes
CLASS_NAMES = [
    'n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02086079-Pekinese',
    'n02086240-Shih-Tzu', 'n02086646-Blenheim_spaniel', 'n02086910-papillon',
    'n02087046-toy_terrier', 'n02087394-Rhodesian_ridgeback', 'n02088094-Afghan_hound',
    'n02088238-basset', 'n02088364-beagle', 'n02088466-bloodhound',
    'n02088632-bluetick', 'n02089078-black-and-tan_coonhound', 'n02089867-Walker_hound',
    'n02089973-English_foxhound', 'n02090379-redbone', 'n02090622-borzoi',
    'n02090721-Irish_wolfhound', 'n02091032-Italian_greyhound', 'n02091134-whippet',
    'n02091244-Ibizan_hound', 'n02091467-Norwegian_elkhound', 'n02091635-otterhound',
    'n02091831-Saluki', 'n02092002-Scottish_deerhound', 'n02092339-Weimaraner',
    'n02093256-Staffordshire_bullterrier', 'n02093428-American_Staffordshire_terrier',
    'n02093647-Bedlington_terrier', 'n02093754-Border_terrier', 'n02093859-Kerry_blue_terrier',
    'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02094258-Norwich_terrier',
    'n02094433-Yorkshire_terrier', 'n02095314-wire-haired_fox_terrier', 'n02095570-Lakeland_terrier',
    'n02095889-Sealyham_terrier', 'n02096051-Airedale', 'n02096177-cairn',
    'n02096294-Australian_terrier', 'n02096437-Dandie_Dinmont', 'n02096585-Boston_bull',
    'n02097047-miniature_schnauzer', 'n02097130-giant_schnauzer', 'n02097209-standard_schnauzer',
    'n02097298-Scotch_terrier', 'n02097474-Tibetan_terrier', 'n02097658-silky_terrier',
    'n02098105-soft-coated_wheaten_terrier', 'n02098286-West_Highland_white_terrier',
    'n02098413-Lhasa', 'n02099267-flat-coated_retriever', 'n02099429-curly-coated_retriever',
    'n02099601-golden_retriever', 'n02099712-Labrador_retriever', 'n02099849-Chesapeake_Bay_retriever',
    'n02100236-German_short-haired_pointer', 'n02100583-vizsla', 'n02100735-English_setter',
    'n02100877-Irish_setter', 'n02101006-Gordon_setter', 'n02101388-Brittany_spaniel',
    'n02101556-clumber', 'n02102040-English_springer', 'n02102177-Welsh_springer_spaniel',
    'n02102318-cocker_spaniel', 'n02102480-Sussex_spaniel', 'n02102973-Irish_water_spaniel',
    'n02104029-kuvasz', 'n02104365-schipperke', 'n02105056-groenendael',
    'n02105162-malinois', 'n02105251-briard', 'n02105412-kelpie',
    'n02105505-komondor', 'n02105641-Old_English_sheepdog', 'n02105855-Shetland_sheepdog',
    'n02106030-collie', 'n02106166-Border_collie', 'n02106382-Bouvier_des_Flandres',
    'n02106550-Rottweiler', 'n02106662-German_shepherd', 'n02107142-Doberman',
    'n02107312-miniature_pinscher', 'n02107574-Greater_Swiss_Mountain_dog', 'n02107683-Bernese_mountain_dog',
    'n02107908-Appenzeller', 'n02108000-EntleBucher', 'n02108089-boxer',
    'n02108422-bull_mastiff', 'n02108551-Tibetan_mastiff', 'n02108915-French_bulldog',
    'n02109047-Great_Dane', 'n02109525-Saint_Bernard', 'n02109961-Eskimo_dog',
    'n02110063-malamute', 'n02110185-Siberian_husky', 'n02110627-affenpinscher',
    'n02110806-basenji', 'n02110958-pug', 'n02111129-Leonberg',
    'n02111277-Newfoundland', 'n02111500-Great_Pyrenees', 'n02111889-Samoyed',
    'n02112018-Pomeranian', 'n02112137-chow', 'n02112350-keeshond',
    'n02112706-Brabancon_griffon', 'n02113023-Pembroke', 'n02113186-Cardigan',
    'n02113624-toy_poodle', 'n02113712-miniature_poodle', 'n02113799-standard_poodle',
    'n02113978-Mexican_hairless', 'n02115641-dingo', 'n02115913-dhole',
    'n02116738-African_hunting_dog'
]

# Display name mapping for user-friendly breed names
BREED_DISPLAY_NAMES = {
    'n02085620-Chihuahua': 'Chihuahua',
    'n02085782-Japanese_spaniel': 'Japanese Spaniel',
    'n02086079-Pekinese': 'Pekinese',
    'n02086240-Shih-Tzu': 'Shih-Tzu',
    'n02086646-Blenheim_spaniel': 'Blenheim Spaniel',
    'n02086910-papillon': 'Papillon',
    'n02087046-toy_terrier': 'Toy Terrier',
    'n02087394-Rhodesian_ridgeback': 'Rhodesian Ridgeback',
    'n02088094-Afghan_hound': 'Afghan Hound',
    'n02088238-basset': 'Basset Hound',
    'n02088364-beagle': 'Beagle',
    'n02088466-bloodhound': 'Bloodhound',
    'n02088632-bluetick': 'Bluetick',
    'n02089078-black-and-tan_coonhound': 'Black-and-tan Coonhound',
    'n02089867-Walker_hound': 'Walker Hound',
    'n02089973-English_foxhound': 'English Foxhound',
    'n02090379-redbone': 'Redbone',
    'n02090622-borzoi': 'Borzoi',
    'n02090721-Irish_wolfhound': 'Irish Wolfhound',
    'n02091032-Italian_greyhound': 'Italian Greyhound',
    'n02091134-whippet': 'Whippet',
    'n02091244-Ibizan_hound': 'Ibizan Hound',
    'n02091467-Norwegian_elkhound': 'Norwegian Elkhound',
    'n02091635-otterhound': 'Otterhound',
    'n02091831-Saluki': 'Saluki',
    'n02092002-Scottish_deerhound': 'Scottish Deerhound',
    'n02092339-Weimaraner': 'Weimaraner',
    'n02093256-Staffordshire_bullterrier': 'Staffordshire Bull Terrier',
    'n02093428-American_Staffordshire_terrier': 'American Staffordshire Terrier',
    'n02093647-Bedlington_terrier': 'Bedlington Terrier',
    'n02093754-Border_terrier': 'Border Terrier',
    'n02093859-Kerry_blue_terrier': 'Kerry Blue Terrier',
    'n02093991-Irish_terrier': 'Irish Terrier',
    'n02094114-Norfolk_terrier': 'Norfolk Terrier',
    'n02094258-Norwich_terrier': 'Norwich Terrier',
    'n02094433-Yorkshire_terrier': 'Yorkshire Terrier',
    'n02095314-wire-haired_fox_terrier': 'Wire-haired Fox Terrier',
    'n02095570-Lakeland_terrier': 'Lakeland Terrier',
    'n02095889-Sealyham_terrier': 'Sealyham Terrier',
    'n02096051-Airedale': 'Airedale',
    'n02096177-cairn': 'Cairn Terrier',
    'n02096294-Australian_terrier': 'Australian Terrier',
    'n02096437-Dandie_Dinmont': 'Dandie Dinmont',
    'n02096585-Boston_bull': 'Boston Bull',
    'n02097047-miniature_schnauzer': 'Miniature Schnauzer',
    'n02097130-giant_schnauzer': 'Giant Schnauzer',
    'n02097209-standard_schnauzer': 'Standard Schnauzer',
    'n02097298-Scotch_terrier': 'Scotch Terrier',
    'n02097474-Tibetan_terrier': 'Tibetan Terrier',
    'n02097658-silky_terrier': 'Silky Terrier',
    'n02098105-soft-coated_wheaten_terrier': 'Soft-coated Wheaten Terrier',
    'n02098286-West_Highland_white_terrier': 'West Highland White Terrier',
    'n02098413-Lhasa': 'Lhasa Apso',
    'n02099267-flat-coated_retriever': 'Flat-coated Retriever',
    'n02099429-curly-coated_retriever': 'Curly-coated Retriever',
    'n02099601-golden_retriever': 'Golden Retriever',
    'n02099712-Labrador_retriever': 'Labrador Retriever',
    'n02099849-Chesapeake_Bay_retriever': 'Chesapeake Bay Retriever',
    'n02100236-German_short-haired_pointer': 'German Short-haired Pointer',
    'n02100583-vizsla': 'Vizsla',
    'n02100735-English_setter': 'English Setter',
    'n02100877-Irish_setter': 'Irish Setter',
    'n02101006-Gordon_setter': 'Gordon Setter',
    'n02101388-Brittany_spaniel': 'Brittany Spaniel',
    'n02101556-clumber': 'Clumber Spaniel',
    'n02102040-English_springer': 'English Springer Spaniel',
    'n02102177-Welsh_springer_spaniel': 'Welsh Springer Spaniel',
    'n02102318-cocker_spaniel': 'Cocker Spaniel',
    'n02102480-Sussex_spaniel': 'Sussex Spaniel',
    'n02102973-Irish_water_spaniel': 'Irish Water Spaniel',
    'n02104029-kuvasz': 'Kuvasz',
    'n02104365-schipperke': 'Schipperke',
    'n02105056-groenendael': 'Groenendael',
    'n02105162-malinois': 'Malinois',
    'n02105251-briard': 'Briard',
    'n02105412-kelpie': 'Kelpie',
    'n02105505-komondor': 'Komondor',
    'n02105641-Old_English_sheepdog': 'Old English Sheepdog',
    'n02105855-Shetland_sheepdog': 'Shetland Sheepdog',
    'n02106030-collie': 'Collie',
    'n02106166-Border_collie': 'Border Collie',
    'n02106382-Bouvier_des_Flandres': 'Bouvier des Flandres',
    'n02106550-Rottweiler': 'Rottweiler',
    'n02106662-German_shepherd': 'German Shepherd',
    'n02107142-Doberman': 'Doberman',
    'n02107312-miniature_pinscher': 'Miniature Pinscher',
    'n02107574-Greater_Swiss_Mountain_dog': 'Greater Swiss Mountain Dog',
    'n02107683-Bernese_mountain_dog': 'Bernese Mountain Dog',
    'n02107908-Appenzeller': 'Appenzeller',
    'n02108000-EntleBucher': 'EntleBucher',
    'n02108089-boxer': 'Boxer',
    'n02108422-bull_mastiff': 'Bull Mastiff',
    'n02108551-Tibetan_mastiff': 'Tibetan Mastiff',
    'n02108915-French_bulldog': 'French Bulldog',
    'n02109047-Great_Dane': 'Great Dane',
    'n02109525-Saint_Bernard': 'Saint Bernard',
    'n02109961-Eskimo_dog': 'Eskimo Dog',
    'n02110063-malamute': 'Malamute',
    'n02110185-Siberian_husky': 'Siberian Husky',
    'n02110627-affenpinscher': 'Affenpinscher',
    'n02110806-basenji': 'Basenji',
    'n02110958-pug': 'Pug',
    'n02111129-Leonberg': 'Leonberg',
    'n02111277-Newfoundland': 'Newfoundland',
    'n02111500-Great_Pyrenees': 'Great Pyrenees',
    'n02111889-Samoyed': 'Samoyed',
    'n02112018-Pomeranian': 'Pomeranian',
    'n02112137-chow': 'Chow',
    'n02112350-keeshond': 'Keeshond',
    'n02112706-Brabancon_griffon': 'Brabancon Griffon',
    'n02113023-Pembroke': 'Pembroke',
    'n02113186-Cardigan': 'Cardigan',
    'n02113624-toy_poodle': 'Toy Poodle',
    'n02113712-miniature_poodle': 'Miniature Poodle',
    'n02113799-standard_poodle': 'Standard Poodle',
    'n02113978-Mexican_hairless': 'Mexican Hairless',
    'n02115641-dingo': 'Dingo',
    'n02115913-dhole': 'Dhole',
    'n02116738-African_hunting_dog': 'African Hunting Dog'
}

# =============================================================================
# PYTORCH MODEL ARCHITECTURE
# =============================================================================

def create_resnet50_model(num_classes=119):
    """
    Create the ResNet50 model architecture used in balanced training.
    
    Creates a ResNet50 model with a custom classifier head for breed classification.
    Base layers are frozen for feature extraction, only the classifier is trainable.
    
    Args:
        num_classes (int): Number of output classes. Default: 119.
        
    Returns:
        nn.Module: Configured ResNet50 model.
    """
    print(f" Creating ResNet50 model for {num_classes} classes...")
    
    # Create pretrained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze base layers for feature extraction
    for param in model.parameters():
        param.requires_grad = False
    
    # Classifier architecture (exact match from training)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    # Only train the classifier
    for param in model.fc.parameters():
        param.requires_grad = True
    
    print(f" ResNet50 model created with {sum(p.numel() for p in model.fc.parameters() if p.requires_grad)} trainable parameters")
    
    return model


def load_best_model():
    """
    Load the best trained model from cross-validation.
    
    Loads the model checkpoint, initializes the architecture, and prepares
    the model for inference on the appropriate device (GPU/CPU).
    
    Returns:
        tuple: (model, device) - Loaded model and computation device.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    # Verify model file exists
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f" Model not found: {MODEL_PATH}")
    
    print(f" Loading model: {MODEL_PATH}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Create model with correct architecture
        model = create_resnet50_model(NUM_CLASSES)
        
        # Checkpoint is either a state_dict directly or a dict with keys
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Dictionary with metadata
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
            accuracy = checkpoint.get('accuracy', 'N/A')
            val_loss = checkpoint.get('val_loss', 'N/A')
        else:
            # Direct state_dict
            model.load_state_dict(checkpoint)
            epoch = 'N/A'
            accuracy = 'N/A'
            val_loss = 'N/A'
        
        model.to(device)
        model.eval()
        
        print(f" Model loaded successfully!")
        print(f" Epoch: {epoch}, Accuracy: {accuracy}, Val Loss: {val_loss}")
        
        return model, device
        
    except Exception as e:
        print(f" Error loading model: {e}")
        raise


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_breed(model, device, image_tensor):
    """
    Perform breed prediction using the loaded model.
    
    Args:
        model (nn.Module): Loaded classification model.
        device (torch.device): Computation device.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        
    Returns:
        list: Top-k predictions with breed, confidence, and threshold info.
    """
    with torch.no_grad():
        # Prepare image
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Prediction
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, TOP_K_PREDICTIONS)
        
        results = []
        for i in range(TOP_K_PREDICTIONS):
            prob = top_probs[0][i].item()
            idx = top_indices[0][i].item()
            class_name = CLASS_NAMES[idx]
            display_name = BREED_DISPLAY_NAMES.get(class_name, class_name.split('-')[-1])
            
            # Apply thresholds adaptativos
            breed_key = class_name.split('-')[-1]
            threshold = ADAPTIVE_THRESHOLDS.get(breed_key, DEFAULT_CLASSIFICATION_THRESHOLD)
            
            results.append({
                'breed': display_name,
                'confidence': prob,
                'class_name': class_name,
                'index': idx,
                'threshold': threshold,
                'is_confident': prob >= threshold
            })
        
        return results

def process_image(image_data):
    """
    Process uploaded image for prediction.
    
    Opens the image, converts to RGB if necessary, and applies the
    required transformations for model input.
    
    Args:
        image_data (bytes): Raw image data from upload.
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
        
    Raises:
        ValueError: If image processing fails.
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image)
        
        return image_tensor
        
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Initialize FastAPI application
app = FastAPI(
    title=" Dog Breed Classifier API - 119 Breeds",
    description="API for dog breed classification using ResNet50 model trained on 119 balanced classes",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS to allow connections from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and device
model = None
device = None


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler to load the model when the API starts.
    
    Called automatically by FastAPI when the server starts.
    Initializes the model and device for inference.
    """
    global model, device
    print(" Starting dog breed classification API...")
    
    try:
        model, device = load_best_model()
        print(" API ready to receive requests!")
    except Exception as e:
        print(f" Initialization error: {e}")
        raise


@app.get("/")
async def root():
    """
    Root endpoint returning API information.
    
    Returns:
        dict: API details including version, model type, and available endpoints.
    """
    return {
        "message": " Dog Breed Classifier API - 119 Breeds",
        "version": "3.0.0",
        "model": "ResNet50 with balanced dataset",
        "classes": NUM_CLASSES,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info",
            "breeds": "/breeds"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and model status.
    
    Returns:
        dict: Health status including model state, device, and GPU availability.
    """
    global model, device
    
    model_loaded = model is not None
    gpu_available = torch.cuda.is_available()
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(device) if device else "unknown",
        "gpu_available": gpu_available,
        "classes": NUM_CLASSES,
        "timestamp": time.time()
    }

@app.get("/model-info")
async def model_info():
    """
    Return detailed model configuration and parameters.
    
    Returns:
        dict: Model architecture, thresholds, and normalization parameters.
    """
    return {
        "model_path": MODEL_PATH,
        "architecture": "ResNet50",
        "num_classes": NUM_CLASSES,
        "input_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "adaptive_thresholds": ADAPTIVE_THRESHOLDS,
        "default_threshold": DEFAULT_CLASSIFICATION_THRESHOLD,
        "top_k": TOP_K_PREDICTIONS
    }

@app.get("/breeds")
async def get_breeds():
    """
    List all breeds that the model can classify.
    
    Returns:
        dict: Total breed count and list with index, class name, display name, and threshold.
    """
    breeds = []
    for i, class_name in enumerate(CLASS_NAMES):
        display_name = BREED_DISPLAY_NAMES.get(class_name, class_name.split('-')[-1])
        breed_key = class_name.split('-')[-1]
        threshold = ADAPTIVE_THRESHOLDS.get(breed_key, DEFAULT_CLASSIFICATION_THRESHOLD)
        
        breeds.append({
            "index": i,
            "class_name": class_name,
            "display_name": display_name,
            "threshold": threshold
        })
    
    return {
        "total_breeds": len(breeds),
        "breeds": breeds
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict dog breed from an uploaded image.
    
    Args:
        file (UploadFile): Image file to classify (JPG, PNG, WEBP).
        
    Returns:
        dict: Classification results with top predictions and confidence scores.
        
    Raises:
        HTTPException: 400 if file is not an image, 500 if model not loaded.
    """
    global model, device
    
    # Verify model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Verify file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        
        # Process image
        start_time = time.time()
        image_tensor = process_image(image_data)
        
        # Perform prediction
        predictions = predict_breed(model, device, image_tensor)
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "success": True,
            "is_dog": True,  # Assuming it's a dog since this is a breed classifier
            "processing_time": round(processing_time, 3),
            "top_predictions": predictions,
            "model_info": {
                "num_classes": NUM_CLASSES,
                "architecture": "ResNet50",
                "device": str(device)
            },
            "recommendation": {
                "most_likely": predictions[0]["breed"] if predictions else "Unknown",
                "confidence": predictions[0]["confidence"] if predictions else 0.0,
                "is_confident": predictions[0]["is_confident"] if predictions else False
            }
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f" Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    print(" Starting API server...")
    print(f" Model: {MODEL_PATH}")
    print(f" Classes: {NUM_CLASSES}")
    print(f" Port: {API_PORT}")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=API_PORT,
        reload=False,
        log_level="info"
    )