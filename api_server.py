"""
API REST for classification of images dog vs NO-dog
Technical documentation in English.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models of respuesta
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    probability: float
    confidence: str
    processing_time_ms: float
    model_version: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    success: bool
    predictions: List[PredictionResponse]
    total_images: int
    total_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str
    uptime_seconds: float

# Implementation note.
app = FastAPI(
    title="🐕 Dog Classification API",
    description="API para clasificación binaria de imágenes: PERRO vs NO-PERRO",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Implementation note.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables global
model_inference = None
model_metadata = None
app_start_time = time.time()
prediction_history = []

async def load_model():
    """Technical documentation in English."""
    global model_inference, model_metadata
    
    try:
        # Search model en directory optimized
        model_dir = Path("./optimized_models")
        model_path = None
        metadata_path = None
        
        # Search files of model
        if (model_dir / "production_model.pt").exists():
            model_path = model_dir / "production_model.pt"
            metadata_path = model_dir / "model_metadata.json"
        elif (model_dir / "production_model.onnx").exists():
            model_path = model_dir / "production_model.onnx"
            metadata_path = model_dir / "model_metadata.json"
        else:
            # Search model original
            best_model_path = Path("./models/best_model.pth")
            if best_model_path.exists():
                logger.warning("Usando modelo original - considera optimizar para producción")
                # Create optimizador temporal
                from inference_optimizer import InferenceOptimizer
                optimizer = InferenceOptimizer(str(best_model_path))
                model_path, metadata_path = optimizer.create_production_model()
        
        if model_path and metadata_path:
            from inference_optimizer import ProductionInference
            model_inference = ProductionInference(str(model_path), str(metadata_path))
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            
            logger.info(f"✅ Modelo cargado exitosamente: {model_path}")
            logger.info(f"   Formato: {model_metadata.get('format', 'unknown')}")
            
        else:
            logger.error("❌ No se encontró ningún modelo")
            model_inference = None
            
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        model_inference = None

@app.on_event("startup")
async def startup_event():
    """Technical documentation in English."""
    logger.info("🚀 Iniciando Dog Classification API...")
    await load_model()
    
    # Create directories necesarios
    Path("./uploads").mkdir(exist_ok=True)
    Path("./temp").mkdir(exist_ok=True)
    
    logger.info("✅ API iniciada correctamente")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Technical documentation in English."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐕 Dog Classification API</title>
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
            <h1>🐕 Dog Classification API</h1>
            <p>Sube una imagen para clasificar si contiene un perro o no</p>
            
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <p>Haz clic aquí o arrastra una imagen</p>
                <input type="file" id="file-input" accept="image/*" style="display: none;">
            </div>
            
            <img id="preview" style="display: none;">
            <br>
            <button id="predict-btn" style="display: none;" onclick="predictImage()">Clasificar Imagen</button>
            
            <div id="result"></div>
            
            <hr>
            <h3>API Endpoints:</h3>
            <ul style="text-align: left;">
                <li><strong>POST /predict</strong> - Clasificar una imagen</li>
                <li><strong>POST /predict/batch</strong> - Clasificar múltiples imágenes</li>
                <li><strong>GET /health</strong> - Estado del servicio</li>
                <li><strong>GET /docs</strong> - Documentación interactiva</li>
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
                
                document.getElementById('result').innerHTML = '<p>Procesando...</p>';
                
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
                                <p>Probabilidad: ${(result.probability * 100).toFixed(1)}%</p>
                                <p>Confianza: ${result.confidence}</p>
                                <p>Tiempo: ${result.processing_time_ms.toFixed(1)} ms</p>
                            </div>
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `
                            <div class="result" style="background-color: # f8d7da;">
                                <p>Error: ${result.error || 'Error desconocido'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `
                        <div class="result" style="background-color: # f8d7da;">
                            <p>Error de conexión: ${error.message}</p>
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
    """Endpoint de salud del servicio"""
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
    """Predice if a image contiene a dog"""
    if not model_inference:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Verify tipo of file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    start_time = time.time()
    
    try:
        # Leer image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Convertir a numpy array
        image_np = np.array(image.convert('RGB'))
        
        # Realizar prediction
        probability, label = model_inference.predict(image_np)
        
        # Calcular confianza
        confidence = "Alta" if abs(probability - 0.5) > 0.3 else "Media" if abs(probability - 0.5) > 0.1 else "Baja"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create respuesta
        prediction_result = PredictionResponse(
            success=True,
            prediction=label,
            probability=float(probability),
            confidence=confidence,
            processing_time_ms=processing_time,
            model_version=model_metadata.get('format', 'unknown') if model_metadata else 'unknown',
            timestamp=datetime.now().isoformat()
        )
        
        # Implementation note.
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
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Technical documentation in English."""
    if not model_inference:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if len(files) > 10:  # Implementation note.
        raise HTTPException(status_code=400, detail="Máximo 10 imágenes por lote")
    
    start_time = time.time()
    predictions = []
    
    try:
        # Procesar all the images
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
        
        # Realizar predictions en lote
        if hasattr(model_inference, 'predict_batch'):
            batch_results = model_inference.predict_batch(images)
        else:
            # Fallback a predictions individuales
            batch_results = [model_inference.predict(img) for img in images]
        
        # Procesar resultados
        for i, (probability, label) in enumerate(batch_results):
            confidence = "Alta" if abs(probability - 0.5) > 0.3 else "Media" if abs(probability - 0.5) > 0.1 else "Baja"
            
            prediction = PredictionResponse(
                success=True,
                prediction=label,
                probability=float(probability),
                confidence=confidence,
                processing_time_ms=0,  # Implementation note.
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
        logger.error(f"Error en predicción por lotes: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imágenes: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Technical documentation in English."""
    if not prediction_history:
        return {"message": "No hay predicciones registradas"}
    
    # Implementation note.
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
        "recent_predictions": prediction_history[-10:]  # Implementation note.
    }

@app.post("/reload-model")
async def reload_model():
    """Technical documentation in English."""
    global model_inference, model_metadata
    
    try:
        await load_model()
        return {"success": True, "message": "Modelo recargado exitosamente"}
    except Exception as e:
        logger.error(f"Error recargando modelo: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

# Middleware for logging of requests
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    return response

if __name__ == "__main__":
    # Configuration for desarrollo
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )