# !/usr/bin/env python3
"""
🚀 server API for balanced model K-FOLD
===============================================
Server FastAPI optimized for servir the best model entrenado
with dataset balanced usando validation cruzada estratificada.
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
# configuration And CONSTANTES
# ====================================================================

# Configuration of the model
MODEL_PATH = "best_model_fold_0.pth"  # best model of the k-fold
NUM_CLASSES = 119  # Classes balanceadas
CONFIDENCE_THRESHOLD = 0.1
TOP_K_PREDICTIONS = 5

# 🚀 thresholds ADAPTATIVOS for CORREGIR false negatives
# Implementation note.
ADAPTIVE_THRESHOLDS = {
    'Lhasa': 0.35,           # Era 46.4% false negatives -> threshold very bajo
    'cairn': 0.40,           # Era 41.4% false negatives -> threshold bajo
    'Siberian_husky': 0.45,  # Era 37.9% false negatives -> threshold bajo-medio
    'whippet': 0.45,         # Era 35.7% false negatives -> threshold bajo-medio
    'malamute': 0.50,        # Era 34.6% false negatives -> threshold medio
    'Australian_terrier': 0.50,  # Era 31.0% false negatives -> threshold medio
    'Norfolk_terrier': 0.50,     # Era 30.8% false negatives -> threshold medio
    'toy_terrier': 0.55,         # Era 30.8% false negatives -> threshold medio-alto
    'Italian_greyhound': 0.55,   # Era 25.9% false negatives -> threshold medio-alto
    'Lakeland_terrier': 0.55,    # Era 24.1% false negatives -> threshold medio-alto
    'bluetick': 0.55,            # Era 24.0% false negatives -> threshold medio-alto
    'Border_terrier': 0.55,      # Era 23.1% false negatives -> threshold medio-alto
    # Breeds normales usan CONFIDENCE_THRESHOLD = 0.1 (very permisivo for top-k)
}

# Threshold by default for classification definitiva
DEFAULT_CLASSIFICATION_THRESHOLD = 0.60

# Transformaciones of image (must match with the of the training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# List of classes balanceadas
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
# model PYTORCH
# ====================================================================

def create_model(n_classes=119):
    """Creates a model ResNet50 for classification (estructura of the k-fold)"""
    model = models.resnet50(pretrained=True)
    
    # Congelar layers base (feature extraction)
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazar clasificador (estructura exacta of the k-fold)
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
    
    # Only entrenar the clasificador
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

# ====================================================================
# Implementation note.
# ====================================================================

app = FastAPI(
    title="🐕 Balanced Dog Breed Classifier API",
    description="API para clasificación de razas de perros usando modelo entrenado con dataset balanceado",
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

# Variables global
model = None
device = None

# ====================================================================
# FUNCIONES of UTILIDAD
# ====================================================================

def load_model():
    """Load the model entrenado"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Usando dispositivo: {device}")
    
    # Search the best model disponible
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
        raise FileNotFoundError("❌ No se encontró ningún modelo entrenado del k-fold")
    
    print(f"📁 Cargando modelo: {selected_model}")
    
    # Create model with the arquitectura correcta of the k-fold
    model = create_model(n_classes=NUM_CLASSES)
    
    try:
        checkpoint = torch.load(selected_model, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"✅ Modelo cargado exitosamente: {selected_model}")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelo {selected_model}: {str(e)}")
        return False

def format_breed_name(class_name: str) -> str:
    """Technical documentation in English."""
    if class_name.startswith('n02'):
        # Implementation note.
        breed_name = class_name.split('-', 1)[1] if '-' in class_name else class_name
        return breed_name.replace('_', ' ').title()
    return class_name.replace('_', ' ').title()

def get_breed_threshold(breed_name: str) -> float:
    """Technical documentation in English."""
    # Implementation note.
    normalized_name = breed_name.lower().replace(' ', '_').replace('-', '_')
    
    # Search en diferentes formatos
    for key in ADAPTIVE_THRESHOLDS:
        if key.lower() == normalized_name or key.lower().replace('_', '') == normalized_name.replace('_', ''):
            return ADAPTIVE_THRESHOLDS[key]
    
    # Implementation note.
    return DEFAULT_CLASSIFICATION_THRESHOLD

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocesa the image for the model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformaciones
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Implementation note.
    return image_tensor

def get_predictions(image_tensor: torch.Tensor, top_k: int = TOP_K_PREDICTIONS) -> Dict:
    """Gets predictions of the model with thresholds adaptativos"""
    global model, device
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Apply softmax for get probabilidades
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get all the predictions for apply thresholds adaptativos
        all_predictions = []
        
        for class_idx in range(len(CLASS_NAMES)):
            confidence = probabilities[class_idx].item()
            class_name = CLASS_NAMES[class_idx]
            breed_name = format_breed_name(class_name)
            
            # Implementation note.
            breed_threshold = get_breed_threshold(breed_name)
            
            # Determinar if supera the threshold (for classification definitiva)
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
        
        # Ordenar for confianza
        all_predictions.sort(key=lambda x: x['raw_confidence'], reverse=True)
        
        # Filtrar predictions that pasan threshold for classification definitiva
        positive_predictions = [p for p in all_predictions if p['passes_threshold']]
        
        # Top-k for show (independiente of threshold)
        top_k_predictions = all_predictions[:top_k]
        
        # Limpiar campos internos of the predictions finales
        final_predictions = []
        for pred in top_k_predictions:
            final_pred = pred.copy()
            del final_pred['raw_confidence']  # No show confianza raw
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
# ENDPOINTS of the API
# ====================================================================

@app.on_event("startup")
async def startup_event():
    """Inicializar the model to the arrancar"""
    success = load_model()
    if not success:
        raise HTTPException(status_code=500, detail="Error cargando el modelo")

@app.get("/")
async def root():
    """Technical documentation in English."""
    return {
        "message": "🐕 Balanced Dog Breed Classifier API",
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
    """Endpoint of salud"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/classes")
async def get_classes():
    """Gets the list of classes disponibles"""
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
    """Clasifica a image of dog"""
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    # Validar tipo of file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer and procesar image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocesar image
        image_tensor = preprocess_image(image)
        
        # Get predictions
        results = get_predictions(image_tensor)
        
        # Implementation note.
        results["image_info"] = {
            "filename": file.filename,
            "size": f"{image.size[0]}x{image.size[1]}",
            "mode": image.mode
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

# ====================================================================
# EJECUTAR server
# ====================================================================

if __name__ == "__main__":
    print("🚀 Iniciando Balanced Dog Breed Classifier API...")
    print("=" * 60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,  # Cambio of port for evitar conflicto
        log_level="info"
    )