"""
API Jerárquica para Detección de Perros y Clasificación de Razas
Sistema de dos etapas optimizado
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
    """Clasificador jerárquico: primero detecta perro, luego clasifica raza"""
    
    def __init__(self, binary_model_path: str, breed_model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Cargar modelo binario (perro vs no-perro)
        print("🔄 Cargando modelo binario...")
        self.binary_model = self.load_binary_model(binary_model_path)
        
        # Cargar modelo de razas (si está disponible)
        self.breed_model = None
        self.breed_names = {}
        
        if breed_model_path and Path(breed_model_path).exists():
            print("🔄 Cargando modelo de razas...")
            self.breed_model = self.load_breed_model(breed_model_path)
        else:
            print("⚠️  Modelo de razas no disponible aún")
        
        # Transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Modelos cargados en dispositivo: {self.device}")
    
    def load_binary_model(self, model_path: str):
        """Carga el modelo binario existente"""
        try:
            # Recrear la arquitectura del modelo binario
            from quick_train import DogClassificationModel
            
            model = DogClassificationModel()
            
            # Cargar pesos
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
        """Carga el modelo de razas"""
        try:
            # Cargar checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Obtener configuración
            model_config = checkpoint.get('model_config', {})
            num_classes = model_config.get('num_classes', 50)
            model_name = model_config.get('model_name', 'efficientnet_b3')
            self.breed_names = model_config.get('breed_names', {})
            
            # Recrear modelo
            from breed_trainer import AdvancedBreedClassifier
            
            model = AdvancedBreedClassifier(
                num_classes=num_classes,
                model_name=model_name,
                pretrained=False
            )
            
            # Cargar pesos
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✅ Modelo de razas cargado: {num_classes} clases")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo de razas: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesa imagen para inferencia"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Aplicar transformaciones
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Agregar batch dimension
        
        return tensor.to(self.device)
    
    def predict_binary(self, image: Image.Image) -> Tuple[bool, float]:
        """Predice si es perro o no (primera etapa)"""
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.binary_model(tensor)
            
            # Si es un modelo binario con sigmoid
            if output.shape[1] == 1:
                prob = torch.sigmoid(output).item()
                is_dog = prob > 0.5
            else:
                # Si es un modelo con 2 clases
                probs = F.softmax(output, dim=1)
                prob = probs[0, 1].item()  # Probabilidad de ser perro
                is_dog = prob > 0.5
        
        return is_dog, prob
    
    def predict_breed(self, image: Image.Image) -> Tuple[str, float, List[Dict]]:
        """Predice la raza del perro (segunda etapa)"""
        if self.breed_model is None:
            return "Modelo de razas no disponible", 0.0, []
        
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.breed_model(tensor)
            probs = F.softmax(output, dim=1)
            
            # Top-5 predicciones
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
            
            # Mejor predicción
            best_breed = predictions[0]['breed']
            best_confidence = predictions[0]['confidence']
            
            return best_breed, best_confidence, predictions
    
    def classify_hierarchical(self, image: Image.Image) -> Dict:
        """Clasificación jerárquica completa"""
        start_time = time.time()
        
        # Etapa 1: ¿Es un perro?
        is_dog, dog_confidence = self.predict_binary(image)
        
        result = {
            'is_dog': is_dog,
            'dog_confidence': dog_confidence,
            'processing_time_ms': 0,
            'breed_info': None
        }
        
        # Etapa 2: Si es perro, ¿qué raza?
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
        
        # Tiempo de procesamiento
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
    """Inicializa el clasificador al arrancar"""
    global classifier
    
    # Rutas de los modelos
    binary_model_path = "best_model.pth"
    breed_model_path = "breed_models/best_breed_model.pth"
    
    # Verificar que existe el modelo binario
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
    """Endpoint raíz con información de la API"""
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
    """Verifica el estado del sistema"""
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
    """Clasificación jerárquica completa"""
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Clasificador no inicializado")
    
    # Validar archivo
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
    try:
        # Leer y procesar imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Clasificar
        result = classifier.classify_hierarchical(image)
        
        # Agregar metadata
        result['filename'] = file.filename
        result['file_size_kb'] = len(image_data) // 1024
        result['image_dimensions'] = image.size
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

@app.post("/detect-dog")
async def detect_dog_only(file: UploadFile = File(...)):
    """Solo detección binaria de perro"""
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
    """Solo clasificación de raza (asume que es un perro)"""
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
    """Lista todas las razas disponibles"""
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
    print("🚀 Iniciando API Jerárquica...")
    print("📡 Endpoints disponibles:")
    print("   http://localhost:8001/classify - Clasificación completa")
    print("   http://localhost:8001/detect-dog - Solo detección de perro")
    print("   http://localhost:8001/classify-breed - Solo clasificación de raza")
    print("   http://localhost:8001/docs - Documentación")
    
    uvicorn.run(
        "hierarchical_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )