"""
API simple for testing of the frontend - Mock API
Simula respuestas de classification de perros
"""

import time
import random
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Implementation note.
BREED_NAMES = [
    "Golden Retriever", "Labrador", "Pastor Alemán", "Bulldog Francés", 
    "Beagle", "Poodle", "Rottweiler", "Yorkshire Terrier", "Siberian Husky",
    "Chihuahua", "Border Collie", "Dachshund", "Boxer", "Shih Tzu",
    "Boston Terrier", "Pomeranian", "Australian Shepherd", "Cocker Spaniel"
]

app = FastAPI(
    title="Dog Classifier Mock API",
    description="API simulada para testing del frontend",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def simulate_dog_detection(filename: str):
    """Technical documentation in English."""
    # Implementation note.
    dog_keywords = ["dog", "perro", "can", "pup", "hund", "chien"]
    not_dog_keywords = ["cat", "gato", "bird", "car", "person", "house"]
    
    filename_lower = filename.lower()
    
    # Verificar palabras clave
    if any(keyword in filename_lower for keyword in dog_keywords):
        return True, random.uniform(0.85, 0.99)
    elif any(keyword in filename_lower for keyword in not_dog_keywords):
        return False, random.uniform(0.05, 0.25)
    else:
        # Implementation note.
        is_dog = random.choice([True, False])
        if is_dog:
            return True, random.uniform(0.70, 0.95)
        else:
            return False, random.uniform(0.10, 0.40)

def simulate_breed_classification():
    """Simula classification de breed"""
    # Seleccionar breed principal
    primary_breed = random.choice(BREED_NAMES)
    primary_confidence = random.uniform(0.60, 0.95)
    
    # Top 5 breeds
    top5_breeds = []
    remaining_breeds = [b for b in BREED_NAMES if b != primary_breed]
    random.shuffle(remaining_breeds)
    
    # Agregar la breed principal
    top5_breeds.append({
        'breed': primary_breed,
        'confidence': round(primary_confidence, 4),
        'class_index': BREED_NAMES.index(primary_breed)
    })
    
    # Implementation note.
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
    """Technical documentation in English."""
    return {
        "service": "Dog Classifier Mock API",
        "version": "1.0.0",
        "status": "active",
        "message": "API simulada para testing del frontend",
        "endpoints": {
            "/classify": "Clasificación completa simulada",
            "/detect": "Solo detección simulada",
            "/health": "Estado del sistema",
            "/docs": "Documentación"
        },
        "note": "Esta es una API mock que simula respuestas para testing"
    }

@app.get("/health")
async def health_check():
    """Estado of the system simulado"""
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
    """Classification completa simulada"""
    
    # Validar file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
    try:
        # Simular tiempo de processing
        start_time = time.time()
        await asyncio.sleep(random.uniform(0.1, 0.3))  # Simular processing
        
        # Leer image for obtener metadatos
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Implementation note.
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
        
        # if it is a dog, simular classification de breed
        if is_dog:
            breed, breed_confidence, top5_breeds = simulate_breed_classification()
            result['breed_info'] = {
                'primary_breed': breed,
                'breed_confidence': round(breed_confidence, 4),
                'top5_breeds': top5_breeds
            }
        
        # Tiempo de processing
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

@app.post("/detect")
async def detect_dog(file: UploadFile = File(...)):
    """Technical documentation in English."""
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
    try:
        start_time = time.time()
        
        # Leer image
        image_data = await file.read()
        
        # Implementation note.
        is_dog, confidence = simulate_dog_detection(file.filename or "unknown")
        
        # Simular tiempo de processing
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
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

# Importar asyncio for sleep
import asyncio

if __name__ == "__main__":
    print("🚀 Iniciando Mock API para Testing...")
    print("📡 Endpoints disponibles:")
    print("   http://localhost:8001/classify - Clasificación simulada")
    print("   http://localhost:8001/detect - Detección simulada")
    print("   http://localhost:8001/health - Estado del sistema")
    print("   http://localhost:8001/docs - Documentación")
    print("🎭 NOTA: Esta es una API MOCK para testing del frontend")
    print("=" * 60)
    
    uvicorn.run(
        "mock_backend_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )