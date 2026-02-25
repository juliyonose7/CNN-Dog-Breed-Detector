"""
API optimizada for trabajar with el model quick_train
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
    title="🐕 Dog Detection API",
    description="API para detectar si hay un perro en una imagen",
    version="1.0.0"
)

# Configurar CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
transform = None
device = torch.device('cpu')

class DogClassificationModel(nn.Module):
    """Model igual to the usado en quick_train.py"""
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1, pretrained: bool = True):
        super(DogClassificationModel, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Cabezal clasificador
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
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()

def load_model():
    """Load el model entrenado"""
    global model, transform
    
    model_path = Path("./quick_models/best_model.pth")
    
    if not model_path.exists():
        print("❌ Modelo no encontrado. Ejecuta primero: python quick_train.py --dataset './DATASETS' --epochs 3")
        return False
    
    try:
        # load model
        checkpoint = torch.load(model_path, map_location=device)
        model = DogClassificationModel(model_name='resnet50')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Transformaciones
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"   Accuracy del modelo: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """load model to the start"""
    print("🚀 Iniciando Dog Detection API...")
    success = load_model()
    if not success:
        print("⚠️  API iniciada sin modelo. Algunas funciones no estarán disponibles.")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Technical documentation in English."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐕 Dog Detection API</title>
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
            <h1>🐕 Dog Detection</h1>
            <p class="subtitle">Inteligencia Artificial para detectar perros en imágenes</p>
            
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <p>📸 Haz clic aquí o arrastra una imagen</p>
                <input type="file" id="file-input" accept="image/*" style="display: none;">
            </div>
            
            <img id="preview" style="display: none;">
            <br>
            <button id="predict-btn" style="display: none;" onclick="predictImage()">🔍 Analizar Imagen</button>
            
            <div id="result"></div>
            
            <div class="api-info">
                <h3>🚀 API Endpoints</h3>
                <ul>
                    <li><strong>POST /predict</strong> - Analizar una imagen</li>
                    <li><strong>GET /health</strong> - Estado del servicio</li>
                    <li><strong>GET /docs</strong> - Documentación completa</li>
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
                
                document.getElementById('result').innerHTML = '<p>🔄 Analizando imagen...</p>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        const isdog = result.class === 'dog';
                        const emoji = isdog ? '🐕' : '📦';
                        const label = isdog ? 'PERRO DETECTADO' : 'NO ES PERRO';
                        
                        document.getElementById('result').innerHTML = `
                            <div class="result ${result.class}">
                                <h2>${emoji} ${label}</h2>
                                <p><strong>Probabilidad:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                                <p><strong>Confianza:</strong> ${result.confidence_level}</p>
                                <p><strong>Tiempo:</strong> ${result.processing_time_ms.toFixed(1)} ms</p>
                            </div>
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `
                            <div class="result no-dog">
                                <p>❌ Error: ${result.error || 'Error desconocido'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `
                        <div class="result no-dog">
                            <p>❌ Error de conexión: ${error.message}</p>
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
    """Predice si una imagen contiene un perro"""
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Implementation note.
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    # Implementation note.
    if not file.content_type:
        filename = file.filename.lower() if file.filename else ""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        if not filename.endswith(valid_extensions):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida")
    
    start_time = time.time()
    
    try:
        # Leer y procesar image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Aplicar transformaciones
        input_tensor = transform(image).unsqueeze(0)
        
        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
        
        # Clasificar
        is_dog = probability > 0.5
        class_name = "dog" if is_dog else "no-dog"
        confidence_level = "Alta" if abs(probability - 0.5) > 0.3 else "Media" if abs(probability - 0.5) > 0.1 else "Baja"
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "class": class_name,  # Compatible with frontend
            "confidence": float(probability),  # Implementation note.
            "confidence_level": confidence_level,  # Texto descriptivo
            "processing_time_ms": processing_time,
            "model_version": "quick_train_v1",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.get("/health")
async def health_check():
    """Estado of the servicio"""
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "device": str(device),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    print("🌐 Iniciando servidor API...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)