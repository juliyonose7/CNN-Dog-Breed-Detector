"""
Mock Backend API for Frontend Testing
======================================

Simulated API server that mimics the behavior of the production dog classification API.
Used for frontend development and testing without requiring actual ML model inference.

Features:
    - Simulates dog detection based on filename keywords
    - Generates realistic breed classification results
    - Mimics processing times and response formats
    - Full CORS support for browser-based testing

Endpoints:
    GET  /          - Service information and available endpoints
    GET  /health    - System health status (simulated)
    POST /classify  - Full classification with breed detection
    POST /detect    - Binary dog detection only

Usage:
    python mock_backend_api.py
    # Server runs on http://localhost:8001

Author: Dog Classification Team
Version: 1.0.0
"""

import time
import random
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Sample breed names for simulation
BREED_NAMES = [
    "Golden Retriever", "Labrador", "German Shepherd", "French Bulldog", 
    "Beagle", "Poodle", "Rottweiler", "Yorkshire Terrier", "Siberian Husky",
    "Chihuahua", "Border Collie", "Dachshund", "Boxer", "Shih Tzu",
    "Boston Terrier", "Pomeranian", "Australian Shepherd", "Cocker Spaniel"
]

app = FastAPI(
    title="Dog Classifier Mock API",
    description="Simulated API for frontend testing",
    version="1.0.0"
)

# Configure CORS to allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def simulate_dog_detection(filename: str):
    """
    Simulate dog detection based on filename keywords.
    
    Uses heuristics based on common words in filenames to produce
    realistic detection results for testing purposes.
    
    Args:
        filename (str): Name of the uploaded image file.
    
    Returns:
        tuple: (is_dog, confidence) where is_dog is boolean and
               confidence is a float between 0 and 1.
    """
    # Keywords indicating dog images
    dog_keywords = ["dog", "perro", "can", "pup", "hund", "chien"]
    not_dog_keywords = ["cat", "gato", "bird", "car", "person", "house"]
    
    filename_lower = filename.lower()
    
    # Check for keywords
    if any(keyword in filename_lower for keyword in dog_keywords):
        return True, random.uniform(0.85, 0.99)
    elif any(keyword in filename_lower for keyword in not_dog_keywords):
        return False, random.uniform(0.05, 0.25)
    else:
        # Random result for ambiguous filenames
        is_dog = random.choice([True, False])
        if is_dog:
            return True, random.uniform(0.70, 0.95)
        else:
            return False, random.uniform(0.10, 0.40)

def simulate_breed_classification():
    """
    Simulate breed classification with realistic confidence distribution.
    
    Generates a primary breed prediction with high confidence and
    additional breeds with decreasing confidence scores.
    
    Returns:
        tuple: (primary_breed, primary_confidence, top5_breeds) where
               top5_breeds is a list of breed prediction dictionaries.
    """
    # Select primary breed
    primary_breed = random.choice(BREED_NAMES)
    primary_confidence = random.uniform(0.60, 0.95)
    
    # Build top 5 breeds list
    top5_breeds = []
    remaining_breeds = [b for b in BREED_NAMES if b != primary_breed]
    random.shuffle(remaining_breeds)
    
    # Add primary breed
    top5_breeds.append({
        'breed': primary_breed,
        'confidence': round(primary_confidence, 4),
        'class_index': BREED_NAMES.index(primary_breed)
    })
    
    # Add remaining breeds with decreasing confidence
    remaining_confidence = 1.0 - primary_confidence
    for i in range(4):
        if i < len(remaining_breeds):
            conf = remaining_confidence * random.uniform(0.1, 0.8) / (i + 1)
            top5_breeds.append({
                'breed': remaining_breeds[i],
                'confidence': round(conf, 4),
                'class_index': BREED_NAMES.index(remaining_breeds[i])
            })
    
    return primary_breed, primary_confidence, top5_breeds

@app.get("/")
async def root():
    """
    Root endpoint with service information.
    
    Returns:
        dict: Service metadata including available endpoints and status.
    """
    return {
        "service": "Dog Classifier Mock API",
        "version": "1.0.0",
        "status": "active",
        "message": "Simulated API for frontend testing",
        "endpoints": {
            "/classify": "Full classification (simulated)",
            "/detect": "Detection only (simulated)",
            "/health": "System status",
            "/docs": "API documentation"
        },
        "note": "This is a mock API that simulates responses for testing"
    }


@app.get("/health")
async def health_check():
    """
    Simulated system health status endpoint.
    
    Returns:
        dict: Health status information (always healthy for mock).
    """
    return {
        "status": "healthy",
        "binary_model_loaded": True,
        "breed_model_loaded": True,
        "device": "cpu",
        "timestamp": time.time(),
        "mock": True
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Simulated full image classification endpoint.
    
    Accepts an image file and returns simulated dog detection and
    breed classification results.
    
    Args:
        file (UploadFile): The image file to classify.
    
    Returns:
        dict: Classification results including dog detection,
              breed predictions, and processing metadata.
    
    Raises:
        HTTPException: 400 if file is not an image, 500 on processing error.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Simulate processing time
        start_time = time.time()
        await asyncio.sleep(random.uniform(0.1, 0.3))  # Simulate processing
        
        # Read image to get metadata
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Simulate detection
        is_dog, dog_confidence = simulate_dog_detection(file.filename or "unknown")
        
        result = {
            'is_dog': is_dog,
            'dog_confidence': round(dog_confidence, 4),
            'prediction': 'dog' if is_dog else 'not_dog',
            'breed_info': None,
            'processing_time_ms': 0,
            'filename': file.filename,
            'file_size_kb': len(image_data) // 1024,
            'image_size': image.size,
            'timestamp': time.time(),
            'mock': True
        }
        
        # If dog detected, simulate breed classification
        if is_dog:
            breed, breed_confidence, top5_breeds = simulate_breed_classification()
            result['breed_info'] = {
                'primary_breed': breed,
                'breed_confidence': round(breed_confidence, 4),
                'top5_breeds': top5_breeds
            }
        
        # Calculate processing time
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")

@app.post("/detect")
async def detect_dog(file: UploadFile = File(...)):
    """
    Simulated dog detection endpoint (binary classification only).
    
    Args:
        file (UploadFile): The image file to analyze.
    
    Returns:
        dict: Detection result with confidence score.
    
    Raises:
        HTTPException: 400 if file is not an image, 500 on processing error.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        start_time = time.time()
        
        # Read image
        image_data = await file.read()
        
        # Simulate detection
        is_dog, confidence = simulate_dog_detection(file.filename or "unknown")
        
        # Simulate processing time
        processing_time = round(random.uniform(50, 200), 2)
        
        return {
            'is_dog': is_dog,
            'confidence': round(confidence, 4),
            'prediction': 'dog' if is_dog else 'not_dog',
            'processing_time_ms': processing_time,
            'filename': file.filename,
            'timestamp': time.time(),
            'mock': True
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")


# Import asyncio for async operations
import asyncio

if __name__ == "__main__":
    print("Starting Mock API for Testing...")
    print("Available endpoints:")
    print("   http://localhost:8001/classify - Simulated classification")
    print("   http://localhost:8001/detect - Simulated detection")
    print("   http://localhost:8001/health - System status")
    print("   http://localhost:8001/docs - API documentation")
    print("NOTE: This is a MOCK API for frontend testing")
    print("=" * 60)
    
    uvicorn.run(
        "mock_backend_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )