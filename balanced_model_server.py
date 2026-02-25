#!/usr/bin/env python3
"""
Balanced K-Fold Model API Server
=================================

FastAPI server optimized for serving the best model trained with
a balanced dataset using stratified cross-validation.

Features:
- ResNet50 backbone with custom classification head
- 119 balanced dog breed classes (252 images per class)
- Adaptive confidence thresholds for problematic breeds
- Top-K predictions with breed-specific optimization
- CORS enabled for cross-origin requests

The server uses adaptive thresholds to reduce false negatives
for breeds that historically showed high false negative rates:
- Lhasa, Cairn, Siberian Husky, Whippet, etc.

Endpoints:
- GET /: API information and endpoints
- GET /health: Health check status
- GET /classes: List of all breed classes
- POST /classify: Image classification endpoint

Author: Dog Breed Classifier Team
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
import numpy as np
from typing import Dict, List
import uvicorn

# ====================================================================
# CONFIGURATION AND CONSTANTS
# ====================================================================

# Model configuration
MODEL_PATH = "best_model_fold_0.pth"  # Best model from k-fold
NUM_CLASSES = 119  # Balanced classes
CONFIDENCE_THRESHOLD = 0.1
TOP_K_PREDICTIONS = 5

# ADAPTIVE THRESHOLDS TO CORRECT FALSE NEGATIVES
# Based on analysis of breeds with highest false negative rates
ADAPTIVE_THRESHOLDS = {
    'Lhasa': 0.35,           # Was 46.4% false negatives -> very low threshold
    'cairn': 0.40,           # Was 41.4% false negatives -> low threshold
    'Siberian_husky': 0.45,  # Was 37.9% false negatives -> low-medium threshold
    'whippet': 0.45,         # Was 35.7% false negatives -> low-medium threshold
    'malamute': 0.50,        # Was 34.6% false negatives -> medium threshold
    'Australian_terrier': 0.50,  # Was 31.0% false negatives -> medium threshold
    'Norfolk_terrier': 0.50,     # Was 30.8% false negatives -> medium threshold
    'toy_terrier': 0.55,         # Was 30.8% false negatives -> medium-high threshold
    'Italian_greyhound': 0.55,   # Was 25.9% false negatives -> medium-high threshold
    'Lakeland_terrier': 0.55,    # Was 24.1% false negatives -> medium-high threshold
    'bluetick': 0.55,            # Was 24.0% false negatives -> medium-high threshold
    'Border_terrier': 0.55,      # Was 23.1% false negatives -> medium-high threshold
    # Normal breeds use CONFIDENCE_THRESHOLD = 0.1 (very permissive for top-k)
}

# Default threshold for definitive classification
DEFAULT_CLASSIFICATION_THRESHOLD = 0.60

# Image transformations (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# List of balanced classes
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

# ====================================================================
# PYTORCH MODEL
# ====================================================================

def create_model(n_classes=119):
    """
    Create a ResNet50 model for classification.
    
    Creates a ResNet50 backbone with a custom classification head
    matching the k-fold training architecture.
    
    Args:
        n_classes (int): Number of output classes. Default is 119.
        
    Returns:
        nn.Module: Configured ResNet50 model.
    """
    model = models.resnet50(pretrained=True)
    
    # Freeze base layers (feature extraction)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier (exact k-fold structure)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, n_classes)
    )
    
    # Only train the classifier
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

# ====================================================================
# FASTAPI APPLICATION
# ====================================================================

app = FastAPI(
    title=" Balanced Dog Breed Classifier API",
    description="API for dog breed classification using a model trained with balanced dataset",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def load_model():
    """
    Load the trained model from checkpoint.
    
    Searches for the best available k-fold model and loads it
    into memory for inference.
    
    Returns:
        bool: True if model loaded successfully, False otherwise.
    """
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    # Search for best available model
    model_files = [
        "best_model_fold_0.pth",
        "best_model_fold_1.pth", 
        "best_model_fold_2.pth",
        "best_model_fold_3.pth",
        "best_model_fold_4.pth"
    ]
    
    selected_model = None
    for model_file in model_files:
        if os.path.exists(model_file):
            selected_model = model_file
            break
    
    if not selected_model:
        raise FileNotFoundError(" No trained k-fold model found")
    
    print(f" Loading model: {selected_model}")
    
    # Create model with correct k-fold architecture
    model = create_model(n_classes=NUM_CLASSES)
    
    try:
        checkpoint = torch.load(selected_model, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f" Model loaded successfully: {selected_model}")
        return True
    except Exception as e:
        print(f" Error loading model {selected_model}: {str(e)}")
        return False

def format_breed_name(class_name: str) -> str:
    """
    Format a class name into a human-readable breed name.
    
    Converts internal class names (like 'n02085620-Chihuahua') to
    display-friendly names ('Chihuahua').
    
    Args:
        class_name (str): Internal class name from the model.
        
    Returns:
        str: Formatted, human-readable breed name.
    """
    if class_name.startswith('n02'):
        # Extract name after the ImageNet synset ID
        breed_name = class_name.split('-', 1)[1] if '-' in class_name else class_name
        return breed_name.replace('_', ' ').title()
    return class_name.replace('_', ' ').title()

def get_breed_threshold(breed_name: str) -> float:
    """
    Get the adaptive confidence threshold for a specific breed.
    
    Returns a lower threshold for breeds with historically high
    false negative rates, or the default threshold otherwise.
    
    Args:
        breed_name (str): Display name of the breed.
        
    Returns:
        float: Confidence threshold for this breed.
    """
    # Normalize the name for lookup
    normalized_name = breed_name.lower().replace(' ', '_').replace('-', '_')
    
    # Search in different formats
    for key in ADAPTIVE_THRESHOLDS:
        if key.lower() == normalized_name or key.lower().replace('_', '') == normalized_name.replace('_', ''):
            return ADAPTIVE_THRESHOLDS[key]
    
    # Return default threshold if breed not in adaptive list
    return DEFAULT_CLASSIFICATION_THRESHOLD

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for model inference.
    
    Converts the image to RGB, applies normalization transforms,
    and adds batch dimension.
    
    Args:
        image (Image.Image): PIL Image to preprocess.
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def get_predictions(image_tensor: torch.Tensor, top_k: int = TOP_K_PREDICTIONS) -> Dict:
    """
    Get model predictions with adaptive thresholds.
    
    Performs inference and returns top-k predictions with
    breed-specific confidence thresholds applied.
    
    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        top_k (int): Number of top predictions to return.
        
    Returns:
        dict: Prediction results including predictions list,
              classification counts, and optimization info.
    """
    global model, device
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get all predictions for adaptive threshold application
        all_predictions = []
        
        for class_idx in range(len(CLASS_NAMES)):
            confidence = probabilities[class_idx].item()
            class_name = CLASS_NAMES[class_idx]
            breed_name = format_breed_name(class_name)
            
            # Get breed-specific threshold
            breed_threshold = get_breed_threshold(breed_name)
            
            # Determine if passes threshold (for definitive classification)
            passes_threshold = confidence >= breed_threshold
            
            all_predictions.append({
                "breed": breed_name,
                "technical_name": class_name,
                "confidence": round(confidence * 100, 2),
                "raw_confidence": confidence,
                "class_id": class_idx,
                "threshold_used": breed_threshold,
                "passes_threshold": passes_threshold,
                "optimization": "OPTIMIZED" if breed_name.lower().replace(' ', '_') in [k.lower() for k in ADAPTIVE_THRESHOLDS.keys()] else "STANDARD"
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['raw_confidence'], reverse=True)
        
        # Filter predictions that pass threshold for definitive classification
        positive_predictions = [p for p in all_predictions if p['passes_threshold']]
        
        # Top-k for display (independent of threshold)
        top_k_predictions = all_predictions[:top_k]
        
        # Clean internal fields from final predictions
        final_predictions = []
        for pred in top_k_predictions:
            final_pred = pred.copy()
            del final_pred['raw_confidence']  # Don't expose raw confidence
            final_predictions.append(final_pred)
        
        return {
            "predictions": final_predictions,
            "positive_classifications": len(positive_predictions),
            "total_classes": len(CLASS_NAMES),
            "model_type": "balanced_kfold_optimized",
            "dataset_info": "29,988 images (252 per class)",
            "optimization_info": {
                "adaptive_thresholds_enabled": True,
                "optimized_breeds": len(ADAPTIVE_THRESHOLDS),
                "false_negative_reduction": "15-25% expected improvement"
            }
        }

# ====================================================================
# API ENDPOINTS
# ====================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on application startup.
    
    Loads the trained model into memory when the server starts.
    Raises HTTPException if model loading fails.
    """
    success = load_model()
    if not success:
        raise HTTPException(status_code=500, detail="Error loading model")

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        dict: API version, model info, and available endpoints.
    """
    return {
        "message": " Balanced Dog Breed Classifier API",
        "version": "2.0.0",
        "model_info": {
            "type": "ResNet50 + Balanced Dataset",
            "classes": NUM_CLASSES,
            "training_method": "5-Fold Stratified Cross Validation",
            "dataset_size": "29,988 images (252 per class)"
        },
        "endpoints": {
            "classify": "/classify",
            "health": "/health",
            "classes": "/classes"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Server status, model state, and device info.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/classes")
async def get_classes():
    """
    Get the list of available breed classes.
    
    Returns:
        dict: Total count and list of all breed classes with IDs.
    """
    formatted_classes = []
    for i, class_name in enumerate(CLASS_NAMES):
        formatted_classes.append({
            "id": i,
            "technical_name": class_name,
            "display_name": format_breed_name(class_name)
        })
    
    return {
        "total_classes": len(CLASS_NAMES),
        "classes": formatted_classes
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded dog image.
    
    Accepts an image file and returns breed predictions with
    confidence scores and adaptive threshold information.
    
    Args:
        file (UploadFile): Image file to classify.
        
    Returns:
        JSONResponse: Prediction results with top-k breeds.
        
    Raises:
        HTTPException: If model not loaded or invalid file type.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Get predictions
        results = get_predictions(image_tensor)
        
        # Add image metadata
        results["image_info"] = {
            "filename": file.filename,
            "size": f"{image.size[0]}x{image.size[1]}",
            "mode": image.mode
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# ====================================================================
# RUN SERVER
# ====================================================================

if __name__ == "__main__":
    print(" Starting Balanced Dog Breed Classifier API...")
    print("=" * 60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,  # Different port to avoid conflict
        log_level="info"
    )