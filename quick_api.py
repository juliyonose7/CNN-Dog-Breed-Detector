"""
Dog Detection REST API Server.

This module provides a FastAPI-based REST API server for dog detection
in images. It serves as the backend for the dog classifier web interface,
handling image uploads and returning predictions from the trained model.

Features:
    - FastAPI server with automatic OpenAPI documentation
    - CORS support for React frontend integration
    - Image validation and preprocessing
    - Binary classification (dog vs no-dog) with confidence scores
    - Interactive HTML interface at root endpoint

Endpoints:
    GET  /        - Interactive web interface for testing
    POST /predict - Submit image for classification
    GET  /health  - Service health check
    GET  /docs    - Swagger API documentation

Usage:
    python quick_api.py
    
    Then access http://localhost:8000 for web interface
    or http://localhost:8000/docs for API documentation.

Dependencies:
    - FastAPI, Uvicorn for web server
    - PyTorch, torchvision for model inference
    - PIL for image processing

Author: AI System
Date: 2024
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import numpy as np
import cv2
from pathlib import Path
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from typing import Optional
import json
from datetime import datetime

app = FastAPI(
    title=" Dog Detection API",
    description="API for detecting whether there is a dog in an image",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and transforms
model = None
transform = None
device = torch.device('cpu')


class DogClassificationModel(nn.Module):
    """
    Dog classification neural network model.
    
    Uses a ResNet50 backbone with a custom classifier head for
    binary classification (dog vs no-dog). Architecture matches
    the model used in quick_train.py for compatibility.
    
    Attributes:
        backbone: ResNet50 feature extractor with identity FC layer.
        classifier: Custom fully-connected classifier head.
    """
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1, pretrained: bool = True):
        """
        Initialize the dog classification model.
        
        Args:
            model_name: Base model architecture. Currently only 'resnet50' supported.
            num_classes: Number of output classes (1 for binary with sigmoid).
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super(DogClassificationModel, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Classifier head with dropout regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W).
        
        Returns:
            Output tensor of shape (batch,) with logits.
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()


def load_model():
    """
    Load the trained model from checkpoint.
    
    Attempts to load the model from the default checkpoint path
    and configures it for inference mode.
    
    Returns:
        bool: True if model loaded successfully, False otherwise.
    """
    global model, transform
    
    model_path = Path("./quick_models/best_model.pth")
    
    if not model_path.exists():
        print(" Model not found. First run: python quick_train.py --dataset './DATASETS' --epochs 3")
        return False
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model = DogClassificationModel(model_name='resnet50')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Configure image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f" Model loaded successfully")
        print(f"   Model accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
        return True
        
    except Exception as e:
        print(f" Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup."""
    print(" Starting Dog Detection API...")
    success = load_model()
    if not success:
        print("  API started without model. Some functions will not be available.")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive web interface for dog detection testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title> Dog Detection API</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, # 1e1e1e, #2d2d2d);
                color: # ffffff;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                text-align: center;
                background: rgba(0,0,0,0.3);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            }
            h1 { 
                color: # 00d4ff;
                text-shadow: 0 0 20px rgba(0,212,255,0.5);
                font-size: 3em;
                margin-bottom: 10px;
            }
            .subtitle {
                color: # 888;
                font-size: 1.2em;
                margin-bottom: 30px;
            }
            .upload-area { 
                border: 2px dashed # 00d4ff;
                padding: 40px; 
                margin: 20px 0;
                border-radius: 15px;
                background: rgba(0,212,255,0.1);
                transition: all 0.3s ease;
            }
            .upload-area:hover { 
                border-color: # ffffff;
                background: rgba(0,212,255,0.2);
                transform: translateY(-2px);
            }
            button { 
                background: linear-gradient(45deg, # 00d4ff, #0099cc);
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer;
                font-size: 1.1em;
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,212,255,0.3);
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,212,255,0.5);
            }
            # preview {
                max-width: 300px; 
                margin: 20px auto;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }
            .result {
                margin: 20px 0;
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .dog { 
                background: linear-gradient(45deg, rgba(0,255,0,0.2), rgba(0,200,0,0.2));
                border: 1px solid # 00ff00;
            }
            .no-dog { 
                background: linear-gradient(45deg, rgba(255,100,100,0.2), rgba(200,0,0,0.2));
                border: 1px solid # ff6464;
            }
            .api-info {
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 20px;
                margin-top: 30px;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Dog Detection</h1>
            <p class="subtitle">Artificial Intelligence for detecting dogs in images</p>
            
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <p> Click here or drag an image</p>
                <input type="file" id="file-input" accept="image/*" style="display: none;">
            </div>
            
            <img id="preview" style="display: none;">
            <br>
            <button id="predict-btn" style="display: none;" onclick="predictImage()"> Analyze Image</button>
            
            <div id="result"></div>
            
            <div class="api-info">
                <h3> API Endpoints</h3>
                <ul>
                    <li><strong>POST /predict</strong> - Analyze an image</li>
                    <li><strong>GET /health</strong> - Service status</li>
                    <li><strong>GET /docs</strong> - Complete documentation</li>
                </ul>
            </div>
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
                
                document.getElementById('result').innerHTML = '<p> Analyzing image...</p>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        const isdog = result.class === 'dog';
                        const emoji = isdog ? '' : '';
                        const label = isdog ? 'DOG DETECTED' : 'NOT A DOG';
                        
                        document.getElementById('result').innerHTML = `
                            <div class="result ${result.class}">
                                <h2>${emoji} ${label}</h2>
                                <p><strong>Probability:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                                <p><strong>Confidence:</strong> ${result.confidence_level}</p>
                                <p><strong>Time:</strong> ${result.processing_time_ms.toFixed(1)} ms</p>
                            </div>
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `
                            <div class="result no-dog">
                                <p> Error: ${result.error || 'Unknown error'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `
                        <div class="result no-dog">
                            <p> Connection error: ${error.message}</p>
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict whether an image contains a dog.
    
    Accepts an uploaded image file and returns classification results
    including class label, confidence score, and processing time.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.).
    
    Returns:
        dict: Prediction results containing:
            - success: Boolean indicating successful processing
            - class: "dog" or "no-dog"
            - confidence: Probability score (0-1)
            - confidence_level: Human-readable confidence (High/Medium/Low)
            - processing_time_ms: Inference time in milliseconds
            - model_version: Version identifier
            - timestamp: ISO format timestamp
    
    Raises:
        HTTPException: 400 if invalid file type, 503 if model unavailable,
                      500 on processing error.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate content type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate file extension as fallback
    if not file.content_type:
        filename = file.filename.lower() if file.filename else ""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        if not filename.endswith(valid_extensions):
            raise HTTPException(status_code=400, detail="File must be a valid image")
    
    start_time = time.time()
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0)
        
        # Run prediction
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
        
        # Classify result
        is_dog = probability > 0.5
        class_name = "dog" if is_dog else "no-dog"
        confidence_level = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.1 else "Low"
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "class": class_name,  # Compatible with frontend
            "confidence": float(probability),  # Probability as float
            "confidence_level": confidence_level,  # Descriptive text
            "processing_time_ms": processing_time,
            "model_version": "quick_train_v1",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Check service health status.
    
    Returns:
        dict: Health status containing model availability,
              device info, and version number.
    """
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "device": str(device),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    print(" Starting API server...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)