"""
API optimizada para clasificación de perros con modelos mejorados
Usar modelos entrenados: binario + razas
Puerto 8001 para conectar con frontend
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
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Usando dispositivo: {self.device}")
        
        # Paths de los mejores modelos
        self.binary_model_path = "enhanced_binary_models/best_model_epoch_1_acc_0.9543.pth"
        self.breed_model_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
        
        # Cargar modelos
        self.binary_model = self.load_binary_model()
        self.breed_model = self.load_breed_model()
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Clasificador optimizado listo!")
    
    def load_binary_model(self):
        """Carga el modelo binario (perro/no-perro)"""
        try:
            if not Path(self.binary_model_path).exists():
                print(f"❌ Modelo binario no encontrado: {self.binary_model_path}")
                return None
            
            print(f"🔄 Cargando modelo binario: {self.binary_model_path}")
            checkpoint = torch.load(self.binary_model_path, map_location=self.device)
            
            # Crear modelo ResNet-18 para clasificación binaria
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)  # 2 clases: perro/no-perro
            
            # Cargar pesos
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✅ Modelo binario cargado exitosamente (Precisión: 95.43%)")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo binario: {e}")
            return None
    
    def load_breed_model(self):
        """Carga el modelo de clasificación de razas"""
        try:
            if not Path(self.breed_model_path).exists():
                print(f"⚠️  Modelo de razas no encontrado: {self.breed_model_path}")
                return None
            
            print(f"🔄 Cargando modelo de razas: {self.breed_model_path}")
            checkpoint = torch.load(self.breed_model_path, map_location=self.device)
            
            # Obtener número de clases del checkpoint
            num_classes = checkpoint.get('num_classes', 50)
            print(f"📊 Número de razas: {num_classes}")
            
            # Crear modelo ResNet-18 para clasificación de razas
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # Cargar pesos
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✅ Modelo de razas cargado exitosamente (Precisión: 91.99%)")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo de razas: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesa imagen para inferencia"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0)  # Agregar batch dimension
        return tensor.to(self.device)
    
    def predict_binary(self, image: Image.Image) -> Tuple[bool, float]:
        """Predice si es perro o no"""
        if self.binary_model is None:
            return False, 0.0
        
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.binary_model(tensor)
            probs = F.softmax(outputs, dim=1)
            dog_prob = probs[0, 1].item()  # Probabilidad de ser perro (clase 1)
            is_dog = dog_prob > 0.5
        
        return is_dog, dog_prob
    
    def predict_breed(self, image: Image.Image) -> Tuple[str, float, List[Dict]]:
        """Predice la raza del perro"""
        if self.breed_model is None:
            return "Modelo de razas no disponible", 0.0, []
        
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.breed_model(tensor)
            probs = F.softmax(outputs, dim=1)
            
            # Top-5 predicciones
            top5_probs, top5_indices = torch.topk(probs, min(5, probs.size(1)))
            
            predictions = []
            for i in range(top5_probs.size(1)):
                class_idx = top5_indices[0, i].item()
                confidence = top5_probs[0, i].item()
                breed_name = f"Raza_{class_idx}"  # Por ahora usar índice, se puede mapear después
                
                predictions.append({
                    'breed': breed_name,
                    'confidence': confidence,
                    'class_index': class_idx
                })
            
            best_breed = predictions[0]['breed']
            best_confidence = predictions[0]['confidence']
            
            return best_breed, best_confidence, predictions
    
    def classify(self, image: Image.Image) -> Dict:
        """Clasificación completa jerárquica"""
        start_time = time.time()
        
        # Paso 1: ¿Es un perro?
        is_dog, dog_confidence = self.predict_binary(image)
        
        result = {
            'is_dog': is_dog,
            'dog_confidence': round(dog_confidence, 4),
            'prediction': 'dog' if is_dog else 'not_dog',
            'processing_time_ms': 0,
            'breed_info': None
        }
        
        # Paso 2: Si es perro, ¿qué raza?
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
                'message': 'Modelo de razas no disponible',
                'status': 'unavailable'
            }
        
        # Tiempo de procesamiento
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return result

# Crear aplicación FastAPI
app = FastAPI(
    title="Dog Classifier API Optimizada",
    description="API para detección y clasificación de perros usando modelos mejorados",
    version="1.0.0"
)

# Configurar CORS para permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar clasificador
classifier = None

@app.on_event("startup")
async def startup_event():
    """Inicializa el clasificador al arrancar la API"""
    global classifier
    try:
        print("🚀 Iniciando API de clasificación de perros...")
        classifier = OptimizedDogClassifier()
        print("✅ API lista para recibir solicitudes!")
    except Exception as e:
        print(f"❌ Error inicializando API: {e}")
        raise HTTPException(500, f"Error inicializando API: {str(e)}")

@app.get("/")
async def root():
    """Información de la API"""
    return {
        "service": "Dog Classifier API Optimizada",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/classify": "Clasificación completa (detección + raza)",
            "/detect": "Solo detección de perro",
            "/health": "Estado del sistema",
            "/docs": "Documentación interactiva"
        },
        "models": {
            "binary": "95.43% precisión",
            "breed": "91.99% precisión"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado del sistema"""
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
    """Clasificación completa: detección + raza"""
    global classifier
    
    if classifier is None:
        raise HTTPException(500, "Clasificador no inicializado")
    
    # Validar archivo
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
    try:
        # Leer imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Clasificar
        result = classifier.classify(image)
        
        # Agregar metadata
        result['filename'] = file.filename
        result['file_size_kb'] = len(image_data) // 1024
        result['image_size'] = image.size
        result['timestamp'] = time.time()
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

@app.post("/detect")
async def detect_dog(file: UploadFile = File(...)):
    """Solo detección de perro (sin clasificación de raza)"""
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
    print("🚀 Iniciando API de Clasificación de Perros...")
    print("📡 Endpoints disponibles:")
    print("   http://localhost:8001/classify - Clasificación completa")
    print("   http://localhost:8001/detect - Solo detección")
    print("   http://localhost:8001/health - Estado del sistema")
    print("   http://localhost:8001/docs - Documentación interactiva")
    print("=" * 60)
    
    uvicorn.run(
        "optimized_backend_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )