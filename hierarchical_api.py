"""
Technical documentation in English.
System of dos etapas optimized
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
    """Technical documentation in English."""
    
    def __init__(self, binary_model_path: str, breed_model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # load model binario (dog vs no-dog)
        print("🔄 Cargando modelo binario...")
        self.binary_model = self.load_binary_model(binary_model_path)
        
        # Implementation note.
        self.breed_model = None
        self.breed_names = {}
        
        if breed_model_path and Path(breed_model_path).exists():
            print("🔄 Cargando modelo de razas...")
            self.breed_model = self.load_breed_model(breed_model_path)
        else:
            print("⚠️  Modelo de razas no disponible aún")
        
        # Transformaciones of image
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Modelos cargados en dispositivo: {self.device}")
    
    def load_binary_model(self, model_path: str):
        """Load the model binario existente"""
        try:
            # Recrear the arquitectura of the model binario
            from quick_train import DogClassificationModel
            
            model = DogClassificationModel()
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            print("✅ Modelo binario cargado exitosamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo binario: {e}")
            raise
    
    def load_breed_model(self, model_path: str):
        """Load the model of breeds"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get configuration
            model_config = checkpoint.get('model_config', {})
            num_classes = model_config.get('num_classes', 50)
            model_name = model_config.get('model_name', 'efficientnet_b3')
            self.breed_names = model_config.get('breed_names', {})
            
            # Recrear model
            from breed_trainer import AdvancedBreedClassifier
            
            model = AdvancedBreedClassifier(
                num_classes=num_classes,
                model_name=model_name,
                pretrained=False
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✅ Modelo de razas cargado: {num_classes} clases")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo de razas: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesa image for inferencia"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformaciones
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def predict_binary(self, image: Image.Image) -> Tuple[bool, float]:
        """Predice if it is a dog or no (primera stage)"""
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.binary_model(tensor)
            
            # If es a model binario with sigmoid
            if output.shape[1] == 1:
                prob = torch.sigmoid(output).item()
                is_dog = prob > 0.5
            else:
                # If es a model with 2 classes
                probs = F.softmax(output, dim=1)
                prob = probs[0, 1].item()  # Probabilidad of ser dog
                is_dog = prob > 0.5
        
        return is_dog, prob
    
    def predict_breed(self, image: Image.Image) -> Tuple[str, float, List[Dict]]:
        """Predice the breed of the dog (segunda stage)"""
        if self.breed_model is None:
            return "Modelo de razas no disponible", 0.0, []
        
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
        """Technical documentation in English."""
        start_time = time.time()
        
        # Stage 1: ¿Es a dog?
        is_dog, dog_confidence = self.predict_binary(image)
        
        result = {
            'is_dog': is_dog,
            'dog_confidence': dog_confidence,
            'processing_time_ms': 0,
            'breed_info': None
        }
        
        # Implementation note.
        if is_dog and self.breed_model is not None:
            breed, breed_confidence, top5_breeds = self.predict_breed(image)
            
            result['breed_info'] = {
                'primary_breed': breed,
                'breed_confidence': breed_confidence,
                'top5_breeds': top5_breeds
            }
        elif is_dog and self.breed_model is None:
            result['breed_info'] = {
                'message': 'Modelo de razas en entrenamiento...',
                'status': 'training'
            }
        
        # Time of processing
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        
        return result

# Inicializar API
app = FastAPI(
    title="API Jerárquica de Clasificación Canina",
    description="Sistema jerárquico: detecta perros y clasifica razas",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar clasificador
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initializes the clasificador to the arrancar"""
    global classifier
    
    # Paths of the models
    binary_model_path = "best_model.pth"
    breed_model_path = "breed_models/best_breed_model.pth"
    
    # Verify that exists the model binario
    if not Path(binary_model_path).exists():
        print(f"❌ Modelo binario no encontrado: {binary_model_path}")
        raise HTTPException(500, "Modelo binario no disponible")
    
    try:
        classifier = HierarchicalDogClassifier(
            binary_model_path=binary_model_path,
            breed_model_path=breed_model_path
        )
        print("🚀 API Jerárquica lista!")
    except Exception as e:
        print(f"❌ Error inicializando clasificador: {e}")
        raise HTTPException(500, f"Error inicializando modelos: {str(e)}")

@app.get("/")
async def root():
    """Technical documentation in English."""
    return {
        "service": "API Jerárquica de Clasificación Canina",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "Detección binaria perro/no-perro",
            "Clasificación de 50 razas caninas",
            "Sistema jerárquico optimizado",
            "Top-5 predicciones de razas"
        ],
        "endpoints": {
            "/classify": "Clasificación jerárquica completa",
            "/detect-dog": "Solo detección de perro",
            "/classify-breed": "Solo clasificación de raza (requiere imagen de perro)",
            "/health": "Estado del sistema"
        }
    }

@app.get("/health")
async def health_check():
    """Verifies the status of the system"""
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
    """Technical documentation in English."""
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Clasificador no inicializado")
    
    # Validar file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
    try:
        # Leer and procesar image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Clasificar
        result = classifier.classify_hierarchical(image)
        
        # Add metadata
        result['filename'] = file.filename
        result['file_size_kb'] = len(image_data) // 1024
        result['image_dimensions'] = image.size
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

@app.post("/detect-dog")
async def detect_dog_only(file: UploadFile = File(...)):
    """Technical documentation in English."""
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Clasificador no inicializado")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
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
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

@app.post("/classify-breed")
async def classify_breed_only(file: UploadFile = File(...)):
    """Only classification of breed (asume that es a dog)"""
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Clasificador no inicializado")
    
    if classifier.breed_model is None:
        raise HTTPException(503, "Modelo de razas no disponible (entrenando...)")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
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
        raise HTTPException(500, f"Error clasificando raza: {str(e)}")

@app.get("/breeds")
async def list_breeds():
    """List all the breeds disponibles"""
    global classifier
    
    if classifier is None or classifier.breed_model is None:
        return {
            'status': 'not_available',
            'message': 'Modelo de razas no disponible',
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