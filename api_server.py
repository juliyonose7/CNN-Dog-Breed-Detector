"""
Dog Classification REST API Server
==================================

A FastAPI-based REST API server for binary image classification (DOG vs NO-DOG).
This module provides HTTP endpoints for single image and batch image predictions,
health monitoring, and statistics tracking.

Features:
    - Single image prediction endpoint
    - Batch image prediction (up to 10 images)
    - Interactive web interface for testing
    - Health monitoring and statistics
    - CORS support for cross-origin requests
    - Request logging middleware

Endpoints:
    - GET /: Interactive web interface
    - GET /health: Service health status
    - POST /predict: Single image classification
    - POST /predict/batch: Batch image classification
    - GET /stats: Prediction statistics
    - POST /reload-model: Reload the ML model

Author: NotDog YesDog Team
Optimized for: AMD 7800X3D
"""

# === IMPORTS ===

# FastAPI framework and middleware
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Data processing and image handling
import numpy as np
import cv2
from pathlib import Path
import asyncio
import aiofiles
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json
import base64
from io import BytesIO
from PIL import Image
import time
import uuid
from pydantic import BaseModel

# === LOGGING CONFIGURATION ===

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === RESPONSE MODELS ===


class PredictionResponse(BaseModel):
    """Response model for single image prediction.
    
    Attributes:
        success: Whether the prediction was successful.
        prediction: The classification result ('DOG' or 'NO-DOG').
        probability: Confidence probability (0.0 to 1.0).
        confidence: Human-readable confidence level ('High', 'Medium', 'Low').
        processing_time_ms: Time taken for prediction in milliseconds.
        model_version: Version/format of the model used.
        timestamp: ISO format timestamp of the prediction.
    """
    success: bool
    prediction: str
    probability: float
    confidence: str
    processing_time_ms: float
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch image predictions.
    
    Attributes:
        success: Whether the batch processing was successful.
        predictions: List of individual prediction responses.
        total_images: Number of images processed.
        total_processing_time_ms: Total time for batch processing in milliseconds.
    """
    success: bool
    predictions: List[PredictionResponse]
    total_images: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check endpoint.
    
    Attributes:
        status: Service health status ('healthy' or 'unhealthy').
        model_loaded: Whether the ML model is loaded and ready.
        model_version: Version/format of the loaded model.
        device: Computing device being used ('GPU' or 'CPU').
        uptime_seconds: Server uptime in seconds.
    """
    status: str
    model_loaded: bool
    model_version: str
    device: str
    uptime_seconds: float


# === FASTAPI APPLICATION SETUP ===

app = FastAPI(
    title=" Dog Classification API",
    description="API for binary image classification: DOG vs NO-DOG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === GLOBAL STATE VARIABLES ===

model_inference = None  # Inference engine instance
model_metadata = None   # Model configuration and metadata
app_start_time = time.time()  # Server start timestamp
prediction_history = []  # Recent predictions cache


# === MODEL LOADING ===


async def load_model():
    """Load and initialize the machine learning model for inference.
    
    Attempts to load the model in the following priority order:
    1. Optimized production model (production_model.pt or .onnx)
    2. Original best model (best_model.pth) with on-the-fly optimization
    
    Updates global model_inference and model_metadata variables.
    
    Raises:
        Exception: If model loading fails (logged but not propagated).
    """
    global model_inference, model_metadata
    
    try:
        # Search for model in optimized models directory
        model_dir = Path("./optimized_models")
        model_path = None
        metadata_path = None
        
        # Search for model files in priority order
        if (model_dir / "production_model.pt").exists():
            model_path = model_dir / "production_model.pt"
            metadata_path = model_dir / "model_metadata.json"
        elif (model_dir / "production_model.onnx").exists():
            model_path = model_dir / "production_model.onnx"
            metadata_path = model_dir / "model_metadata.json"
        else:
            # Fall back to original unoptimized model
            best_model_path = Path("./models/best_model.pth")
            if best_model_path.exists():
                logger.warning("Using original model - consider optimizing for production")
                # Create temporary optimizer to prepare production model
                from inference_optimizer import InferenceOptimizer
                optimizer = InferenceOptimizer(str(best_model_path))
                model_path, metadata_path = optimizer.create_production_model()
        
        if model_path and metadata_path:
            from inference_optimizer import ProductionInference
            model_inference = ProductionInference(str(model_path), str(metadata_path))
            
            # Load model metadata configuration
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            
            logger.info(f" Model loaded successfully: {model_path}")
            logger.info(f"   Format: {model_metadata.get('format', 'unknown')}")
            
        else:
            logger.error(" No model found")
            model_inference = None
            
    except Exception as e:
        logger.error(f" Error loading model: {e}")
        model_inference = None

# === LIFECYCLE EVENTS ===


@app.on_event("startup")
async def startup_event():
    """Handle application startup tasks.
    
    Performs initialization tasks when the server starts:
    - Logs startup message
    - Loads the ML model into memory
    - Creates required directories for uploads and temporary files
    """
    logger.info(" Starting Dog Classification API...")
    await load_model()
    
    # Create necessary directories for file handling
    Path("./uploads").mkdir(exist_ok=True)
    Path("./temp").mkdir(exist_ok=True)
    
    logger.info(" API started successfully")

# === API ENDPOINTS ===


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive web interface for image classification.
    
    Returns:
        HTMLResponse: A complete HTML page with:
            - File upload functionality
            - Image preview
            - Classification button
            - Results display
            - API documentation links
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title> Dog Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-area { border: 2px dashed # ccc; padding: 40px; margin: 20px 0; }
            .upload-area:hover { border-color: # 007bff; }
            .result { margin: 20px 0; padding: 20px; border-radius: 5px; }
            .dog { background-color: # d4edda; border: 1px solid #c3e6cb; }
            .no-dog { background-color: # f8d7da; border: 1px solid #f5c6cb; }
            button { background-color: # 007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: # 0056b3; }
            # preview { max-width: 300px; margin: 20px auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Dog Classification API</h1>
            <p>Upload an image to classify if it contains a dog or not</p>
            
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <p>Click here or drag and drop an image</p>
                <input type="file" id="file-input" accept="image/*" style="display: none;">
            </div>
            
            <img id="preview" style="display: none;">
            <br>
            <button id="predict-btn" style="display: none;" onclick="predictImage()">Classify Image</button>
            
            <div id="result"></div>
            
            <hr>
            <h3>API Endpoints:</h3>
            <ul style="text-align: left;">
                <li><strong>POST /predict</strong> - Classify a single image</li>
                <li><strong>POST /predict/batch</strong> - Classify multiple images</li>
                <li><strong>GET /health</strong> - Service status</li>
                <li><strong>GET /docs</strong> - Interactive documentation</li>
            </ul>
        </div>

        <script>
            let selectedFile = null;
            
            document.getElementById('file-input').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const preview = document.getElementById('preview');
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                        document.getElementById('predict-btn').style.display = 'inline-block';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            async function predictImage() {
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                document.getElementById('result').innerHTML = '<p>Processing...</p>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        const resultClass = result.prediction.includes('PERRO') ? 'dog' : 'no-dog';
                        document.getElementById('result').innerHTML = `
                            <div class="result ${resultClass}">
                                <h3>${result.prediction}</h3>
                                <p>Probability: ${(result.probability * 100).toFixed(1)}%</p>
                                <p>Confidence: ${result.confidence}</p>
                                <p>Time: ${result.processing_time_ms.toFixed(1)} ms</p>
                            </div>
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `
                            <div class="result" style="background-color: # f8d7da;">
                                <p>Error: ${result.error || 'Unknown error'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `
                        <div class="result" style="background-color: # f8d7da;">
                            <p>Connection error: ${error.message}</p>
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the API service.
    
    Returns:
        HealthResponse: Current service health information including:
            - Service status (healthy/unhealthy)
            - Model loading state
            - Model version
            - Computing device (CPU/GPU)
            - Server uptime
    """
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy" if model_inference else "unhealthy",
        model_loaded=model_inference is not None,
        model_version=model_metadata.get('format', 'unknown') if model_metadata else 'unknown',
        device="GPU" if model_inference and hasattr(model_inference, 'device') and 'cuda' in str(model_inference.device) else "CPU",
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Classify a single image as DOG or NO-DOG.
    
    Args:
        file: The image file to classify (JPEG, PNG, etc.).
    
    Returns:
        PredictionResponse: Classification result with probability and confidence.
    
    Raises:
        HTTPException: 503 if model is not available, 400 if file is not an image,
                      500 for processing errors.
    """
    if not model_inference:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Verify file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    
    try:
        # Read and decode image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Convert to numpy array for model inference
        image_np = np.array(image.convert('RGB'))
        
        # Perform prediction
        probability, label = model_inference.predict(image_np)
        
        # Calculate confidence level based on probability distance from threshold
        confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.1 else "Low"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build response object
        prediction_result = PredictionResponse(
            success=True,
            prediction=label,
            probability=float(probability),
            confidence=confidence,
            processing_time_ms=processing_time,
            model_version=model_metadata.get('format', 'unknown') if model_metadata else 'unknown',
            timestamp=datetime.now().isoformat()
        )
        
        # Store prediction in history for statistics (keep last 100)
        prediction_history.append({
            'filename': file.filename,
            'prediction': label,
            'probability': float(probability),
            'timestamp': datetime.now().isoformat()
        })
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Classify multiple images in a single batch request.
    
    Args:
        files: List of image files to classify (max 10 images).
    
    Returns:
        BatchPredictionResponse: List of individual predictions with total processing time.
    
    Raises:
        HTTPException: 503 if model unavailable, 400 if too many images,
                      500 for processing errors.
    """
    if not model_inference:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if len(files) > 10:  # Limit batch size for performance
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    start_time = time.time()
    predictions = []
    
    try:
        # Process all valid images
        images = []
        filenames = []
        
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
            
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            image_np = np.array(image.convert('RGB'))
            images.append(image_np)
            filenames.append(file.filename)
        
        # Perform batch predictions
        if hasattr(model_inference, 'predict_batch'):
            batch_results = model_inference.predict_batch(images)
        else:
            # Fallback to individual predictions if batch method unavailable
            batch_results = [model_inference.predict(img) for img in images]
        
        # Process and format results
        for i, (probability, label) in enumerate(batch_results):
            confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.1 else "Low"
            
            prediction = PredictionResponse(
                success=True,
                prediction=label,
                probability=float(probability),
                confidence=confidence,
                processing_time_ms=0,  # Individual times not tracked in batch mode
                model_version=model_metadata.get('format', 'unknown') if model_metadata else 'unknown',
                timestamp=datetime.now().isoformat()
            )
            predictions.append(prediction)
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            success=True,
            predictions=predictions,
            total_images=len(predictions),
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get prediction statistics and recent history.
    
    Returns:
        dict: Statistics including total predictions, dog/no-dog counts,
              percentages, average probability, uptime, and recent predictions.
    """
    if not prediction_history:
        return {"message": "No predictions recorded"}
    
    # Calculate prediction statistics
    total_predictions = len(prediction_history)
    dog_predictions = sum(1 for p in prediction_history if 'PERRO' in p['prediction'])
    no_dog_predictions = total_predictions - dog_predictions
    
    avg_probability = np.mean([p['probability'] for p in prediction_history])
    
    return {
        "total_predictions": total_predictions,
        "dog_predictions": dog_predictions,
        "no_dog_predictions": no_dog_predictions,
        "dog_percentage": (dog_predictions / total_predictions * 100) if total_predictions > 0 else 0,
        "average_probability": float(avg_probability),
        "uptime_seconds": time.time() - app_start_time,
        "recent_predictions": prediction_history[-10:]  # Return last 10 predictions
    }


@app.post("/reload-model")
async def reload_model():
    """Reload the machine learning model from disk.
    
    Use this endpoint to pick up model updates without restarting the server.
    
    Returns:
        dict: Success status and message.
    """
    global model_inference, model_metadata
    
    try:
        await load_model()
        return {"success": True, "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


# === MIDDLEWARE ===


@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all incoming HTTP requests with timing information."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    return response


# === MAIN ENTRY POINT ===


if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )