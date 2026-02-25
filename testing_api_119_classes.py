#!/usr/bin/env python3
"""
api servidor para testing del mejor modelo resnet50 con 119 clases
servidor fastapi optimizado que sirve el modelo entrenado de clasificacion
de razas de perros con dataset balanceado y umbrales adaptativos

caracteristicas principales:
- carga modelo best_model_fold_0.pth entrenado con k-fold validation
- maneja 119 clases de razas balanceadas
- implementa umbrales adaptativos para mejorar precision
- api rest con endpoints para prediccion e informacion del modelo
- soporte para imagenes jpg, png, webp
- respuestas json con top-k predicciones y niveles de confianza
"""

# imports del sistema operativo y manejo de archivos
import os          # operaciones del sistema operativo
import time        # medicion de tiempo de procesamiento
import json        # manejo de datos json
from pathlib import Path  # manejo moderno de rutas
from typing import Dict, List, Optional  # anotaciones de tipos

# imports de pytorch para deep learning
import torch                           # framework principal
import torch.nn as nn                  # modulos de redes neuronales
import torchvision.transforms as transforms  # transformaciones de imagenes
import torchvision.models as models    # modelos preentrenados

# imports de fastapi para api web
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # soporte cors
from fastapi.responses import JSONResponse          # respuestas json

# imports de procesamiento de imagenes
from PIL import Image    # biblioteca de imagenes python
import io               # operaciones de entrada/salida
import numpy as np      # operaciones numericas
import uvicorn          # servidor asgi para fastapi

# configuracion principal del modelo y servidor api
# estas constantes controlan el comportamiento de la clasificacion

# ruta al archivo del modelo entrenado con mejor rendimiento
MODEL_PATH = "best_model_fold_0.pth"  # mejor modelo del k-fold cross validation

# numero total de clases que puede clasificar el modelo
NUM_CLASSES = 119  # razas balanceadas en el dataset final

# umbral minimo de confianza para considerar una prediccion valida
CONFIDENCE_THRESHOLD = 0.1  # muy bajo para capturar todas las posibilidades

# numero de mejores predicciones a devolver en la respuesta
TOP_K_PREDICTIONS = 5  # suficiente para mostrar alternativas al usuario

# puerto donde se ejecuta el servidor api
API_PORT = 8000  # puerto estandar para desarrollo

# sistema de umbrales adaptativos para mejorar precision por raza
# cada raza tiene su propio umbral basado en su rendimiento historico
# umbrales mas altos = mayor exigencia para esa raza especifica
ADAPTIVE_THRESHOLDS = {
    'Lhasa': 0.35,                # lhasa apso requiere 35% minimo
    'cairn': 0.40,                # cairn terrier requiere 40% minimo
    'Siberian_husky': 0.45,       # husky siberiano requiere 45% minimo
    'whippet': 0.45,              # whippet requiere 45% minimo
    'malamute': 0.50,             # malamute requiere 50% minimo
    'Australian_terrier': 0.50,   # terrier australiano requiere 50% minimo
    'Norfolk_terrier': 0.50,      # norfolk terrier requiere 50% minimo
    'toy_terrier': 0.55,          # toy terrier requiere 55% minimo
    'Italian_greyhound': 0.55,    # galgo italiano requiere 55% minimo
    'Lakeland_terrier': 0.55,     # lakeland terrier requiere 55% minimo
    'bluetick': 0.55,             # bluetick coonhound requiere 55% minimo
    'Border_terrier': 0.55,      
}

# Threshold por defecto para clasificación definitiva
DEFAULT_CLASSIFICATION_THRESHOLD = 0.60

# Transformaciones de imagen (deben coincidir con las del entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lista completa de 119 clases balanceadas
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

# Mapeo para nombres más legibles
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

# ====================================================================
# MODELO PYTORCH
# ====================================================================

def create_resnet50_model(num_classes=119):
    """
    Crea el modelo ResNet50 exacto usado en el entrenamiento balanceado
    """
    print(f"🏗️ Creando modelo ResNet50 para {num_classes} clases...")
    
    # Crear ResNet50 pretrained
    model = models.resnet50(pretrained=True)
    
    # Congelar capas base para feature extraction
    for param in model.parameters():
        param.requires_grad = False
    
    # Arquitectura del clasificador (exacta del entrenamiento)
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
    
    # Solo entrenar el clasificador
    for param in model.fc.parameters():
        param.requires_grad = True
    
    print(f"✅ Modelo ResNet50 creado con {sum(p.numel() for p in model.fc.parameters() if p.requires_grad)} parámetros entrenables")
    
    return model

def load_best_model():
    """
    Carga el mejor modelo entrenado con validación cruzada
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Usando dispositivo: {device}")
    
    # Verificar que el modelo existe
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"❌ Modelo no encontrado: {MODEL_PATH}")
    
    print(f"🔄 Cargando modelo: {MODEL_PATH}")
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Crear modelo con la arquitectura correcta
        model = create_resnet50_model(NUM_CLASSES)
        
        # El checkpoint es directamente un state_dict, no un diccionario con claves
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Si es un diccionario con claves
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
            accuracy = checkpoint.get('accuracy', 'N/A')
            val_loss = checkpoint.get('val_loss', 'N/A')
        else:
            # Si es directamente un state_dict
            model.load_state_dict(checkpoint)
            epoch = 'N/A'
            accuracy = 'N/A'
            val_loss = 'N/A'
        
        model.to(device)
        model.eval()
        
        print(f"✅ Modelo cargado exitosamente!")
        print(f"📊 Época: {epoch}, Precisión: {accuracy}, Val Loss: {val_loss}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise

# ====================================================================
# FUNCIONES DE PREDICCIÓN
# ====================================================================

def predict_breed(model, device, image_tensor):
    """
    Realiza predicción de raza con el modelo cargado
    """
    with torch.no_grad():
        # Preparar imagen
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Predicción
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Obtener top-k predicciones
        top_probs, top_indices = torch.topk(probabilities, TOP_K_PREDICTIONS)
        
        results = []
        for i in range(TOP_K_PREDICTIONS):
            prob = top_probs[0][i].item()
            idx = top_indices[0][i].item()
            class_name = CLASS_NAMES[idx]
            display_name = BREED_DISPLAY_NAMES.get(class_name, class_name.split('-')[-1])
            
            # Aplicar umbrales adaptativos
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
    Procesa la imagen subida para predicción
    """
    try:
        # Abrir imagen
        image = Image.open(io.BytesIO(image_data))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Aplicar transformaciones
        image_tensor = transform(image)
        
        return image_tensor
        
    except Exception as e:
        raise ValueError(f"Error procesando imagen: {e}")

# ====================================================================
# APLICACIÓN FASTAPI
# ====================================================================

# Inicializar aplicación
app = FastAPI(
    title="🐕 Dog Breed Classifier API - 119 Breeds",
    description="API para clasificación de razas de perros con modelo ResNet50 entrenado en 119 clases balanceadas",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo
model = None
device = None

@app.on_event("startup")
async def startup_event():
    """
    Cargar modelo al iniciar la aplicación
    """
    global model, device
    print("🚀 Iniciando API de clasificación de razas de perros...")
    
    try:
        model, device = load_best_model()
        print("✅ API lista para recibir peticiones!")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        raise

@app.get("/")
async def root():
    """
    Endpoint raíz con información de la API
    """
    return {
        "message": "🐕 Dog Breed Classifier API - 119 Breeds",
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
    Verificar estado de la API y modelo
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
    Información detallada del modelo
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
    Lista de todas las razas que puede clasificar el modelo
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
    Predecir raza de perro desde imagen subida
    """
    global model, device
    
    # Verificar que el modelo está cargado
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    # Verificar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer imagen
        image_data = await file.read()
        
        # Procesar imagen
        start_time = time.time()
        image_tensor = process_image(image_data)
        
        # Realizar predicción
        predictions = predict_breed(model, device, image_tensor)
        processing_time = time.time() - start_time
        
        # Preparar respuesta
        response = {
            "success": True,
            "is_dog": True,  # Asumiendo que es perro ya que es un clasificador de razas
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
        print(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

if __name__ == "__main__":
    print("🚀 Iniciando servidor API...")
    print(f"📊 Modelo: {MODEL_PATH}")
    print(f"🔢 Clases: {NUM_CLASSES}")
    print(f"🌐 Puerto: {API_PORT}")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=API_PORT,
        reload=False,
        log_level="info"
    )