#!/usr/bin/env python3
"""
🐕 CLASIFICADOR JERÁRQUICO DE PERROS - SISTEMA INTEGRADO
========================================================

Combina dos modelos entrenados:
1. Modelo Binario (ResNet18): Detecta si es perro o no
2. Modelo de Razas (ResNet34): Identifica entre 50 razas

Arquitecturas diferentes manejadas correctamente.
Incluye API Flask y frontend web interactivo.

Autor: Sistema IA
Fecha: 2024
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import base64
import io
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DEFINICIÓN DE MODELOS (diferentes arquitecturas)
# =============================================================================

class FastBinaryModel(nn.Module):
    """Modelo binario basado en ResNet18"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class BreedModel(nn.Module):
    """Modelo de razas basado en ResNet34"""
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# CLASIFICADOR JERÁRQUICO PRINCIPAL
# =============================================================================

class HierarchicalDogClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Dispositivo: {self.device}")
        
        # Modelos
        self.binary_model = None
        self.breed_model = None
        self.selective_model = None
        self.selective_classes = {}
        self.selective_idx_to_breed = {}
        
        # Clases
        self.binary_classes = ['nodog', 'dog']
        self.breed_classes = []
        
        # Temperature Scaling para suavizar predicciones
        self.breed_temperature = 10.0  # Temperatura óptima encontrada
        self.binary_temperature = 1.0  # Mantener binario normal
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar modelos
        self._load_models()
        
    def _load_models(self):
        """Carga ambos modelos entrenados"""
        try:
            # 1. MODELO BINARIO (ResNet18)
            binary_path = "realtime_binary_models/best_model_epoch_1_acc_0.9649.pth"
            if os.path.exists(binary_path):
                logger.info("📁 Cargando modelo binario (ResNet18)...")
                self.binary_model = FastBinaryModel(num_classes=2).to(self.device)
                checkpoint = torch.load(binary_path, map_location=self.device)
                self.binary_model.load_state_dict(checkpoint['model_state_dict'])
                self.binary_model.eval()
                logger.info(f"✅ Modelo binario cargado - Accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
            else:
                logger.error(f"❌ Modelo binario no encontrado: {binary_path}")
                
            # 2. MODELO DE RAZAS (ResNet34) - VERSIÓN BALANCEADA
            breed_path = "balanced_models/best_balanced_breed_model_epoch_20_acc_88.1366.pth"
            if os.path.exists(breed_path):
                logger.info("📁 Cargando modelo de razas BALANCEADO (ResNet50)...")
                
                # Definir modelo balanceado (ResNet50)
                class BalancedBreedClassifier(nn.Module):
                    def __init__(self, num_classes=50):
                        super().__init__()
                        self.backbone = models.resnet50(weights=None)
                        num_ftrs = self.backbone.fc.in_features
                        self.backbone.fc = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(num_ftrs, 512),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(512, num_classes)
                        )
                    def forward(self, x):
                        return self.backbone(x)
                
                self.breed_model = BalancedBreedClassifier(num_classes=50).to(self.device)
                checkpoint = torch.load(breed_path, map_location=self.device)
                self.breed_model.load_state_dict(checkpoint['model_state_dict'])
                self.breed_model.eval()
                logger.info(f"✅ Modelo de razas BALANCEADO cargado - Accuracy: {checkpoint.get('val_accuracy', 0):.2f}%")
                logger.info(f"📊 Dataset balanceado: {checkpoint.get('images_per_class', 0)} imágenes por clase")
                
                # Cargar nombres de razas
                self._load_breed_names()
            else:
                logger.warning(f"⚠️ Modelo balanceado no encontrado, intentando modelo original...")
                # Fallback al modelo original
                breed_path_original = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
                if os.path.exists(breed_path_original):
                    logger.info("📁 Cargando modelo de razas original (ResNet34)...")
                    self.breed_model = BreedModel(num_classes=50).to(self.device)
                    checkpoint = torch.load(breed_path_original, map_location=self.device)
                    self.breed_model.load_state_dict(checkpoint['model_state_dict'])
                    self.breed_model.eval()
                    logger.info(f"✅ Modelo de razas original cargado - Accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
                    self._load_breed_names()
                else:
                    logger.error(f"❌ Ni modelo balanceado ni original encontrados")
                
            # 3. MODELO SELECTIVO (Razas problemáticas)
            self.selective_model = None
            self.selective_classes = {}
            selective_path = "selective_models/best_selective_model.pth"
            
            if os.path.exists(selective_path):
                logger.info("📁 Cargando modelo selectivo (6 razas problemáticas)...")
                
                # Definir modelo selectivo
                class SelectiveBreedClassifier(nn.Module):
                    def __init__(self, num_classes):
                        super().__init__()
                        self.backbone = models.resnet34(weights=None)
                        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
                    def forward(self, x):
                        return self.backbone(x)
                
                checkpoint = torch.load(selective_path, map_location=self.device)
                self.selective_model = SelectiveBreedClassifier(6).to(self.device)
                self.selective_model.load_state_dict(checkpoint['model_state_dict'])
                self.selective_model.eval()
                
                # Mapeo de clases selectivas
                self.selective_classes = checkpoint['class_to_idx']
                self.selective_idx_to_breed = {v: k for k, v in self.selective_classes.items()}
                
                logger.info(f"✅ Modelo selectivo cargado - Accuracy: {checkpoint.get('val_accuracy', 0):.2f}%")
                logger.info(f"📋 Razas problemáticas: {list(self.selective_classes.keys())}")
            else:
                logger.warning("⚠️ Modelo selectivo no encontrado, usando solo modelo principal")
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            
    def _load_breed_names(self):
        """Carga los nombres de las 50 razas"""
        breed_data_path = "breed_processed_data/train"
        if os.path.exists(breed_data_path):
            self.breed_classes = sorted([d for d in os.listdir(breed_data_path) 
                                       if os.path.isdir(os.path.join(breed_data_path, d))])
            logger.info(f"📋 Cargadas {len(self.breed_classes)} razas")
        else:
            logger.warning("⚠️ Directorio de razas no encontrado, usando nombres genéricos")
            self.breed_classes = [f"Raza_{i:02d}" for i in range(50)]
    
    def predict_image(self, image_path_or_pil, confidence_threshold=0.5):
        """
        Clasificación jerárquica completa
        
        Args:
            image_path_or_pil: Path a imagen o objeto PIL
            confidence_threshold: Umbral de confianza para clasificación binaria
            
        Returns:
            dict con resultados completos
        """
        try:
            # Cargar y procesar imagen
            if isinstance(image_path_or_pil, str):
                image = Image.open(image_path_or_pil).convert('RGB')
            else:
                image = image_path_or_pil.convert('RGB')
                
            # Transformar imagen
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'image_size': image.size,
                'is_dog': False,
                'binary_confidence': 0.0,
                'breed': None,
                'breed_confidence': 0.0,
                'breed_top3': [],
                'temperature': self.breed_temperature,
                'error': None
            }
            
            # PASO 1: CLASIFICACIÓN BINARIA
            if self.binary_model is not None:
                logger.info("🔍 Iniciando clasificación binaria...")
                with torch.no_grad():
                    binary_output = self.binary_model(input_tensor)
                    # Aplicar temperature scaling al modelo binario
                    binary_probs = F.softmax(binary_output / self.binary_temperature, dim=1)
                    binary_confidence, binary_pred = torch.max(binary_probs, 1)
                    
                    results['binary_confidence'] = float(binary_confidence.item())
                    results['is_dog'] = bool(binary_pred.item() == 1)  # 1 = dog
                    
                    logger.info(f"🔍 Binario: {'PERRO' if results['is_dog'] else 'NO PERRO'} "
                              f"(confianza: {results['binary_confidence']:.4f})")
            else:
                logger.error("❌ Modelo binario es None!")
                results['error'] = "Modelo binario no disponible"
                return results
            
            # PASO 2: CLASIFICACIÓN DE RAZA (solo si es perro)
            if results['is_dog'] and results['binary_confidence'] >= confidence_threshold:
                logger.info(f"🐕 Iniciando clasificación de razas (confianza: {results['binary_confidence']:.4f} >= {confidence_threshold})")
                if self.breed_model is not None and self.breed_classes:
                    logger.info(f"🐕 Modelo de razas disponible, {len(self.breed_classes)} razas cargadas")
                    with torch.no_grad():
                        breed_output = self.breed_model(input_tensor)
                        # Aplicar temperature scaling para suavizar predicciones de razas
                        breed_probs = F.softmax(breed_output / self.breed_temperature, dim=1)
                        
                        # Top-1 predicción principal
                        breed_confidence, breed_pred = torch.max(breed_probs, 1)
                        main_breed = self.breed_classes[breed_pred.item()]
                        main_confidence = float(breed_confidence.item())
                        
                        # === SISTEMA HÍBRIDO: USAR MODELO SELECTIVO PARA RAZAS PROBLEMÁTICAS ===
                        # Verificar si la predicción principal es una raza problemática
                        problematic_breeds = ['basset', 'beagle', 'Labrador_retriever', 'Norwegian_elkhound', 'pug', 'Samoyed']
                        use_selective = False
                        
                        if self.selective_model is not None and main_breed in problematic_breeds:
                            logger.info(f"🎯 Raza problemática detectada: {main_breed}, usando modelo selectivo...")
                            use_selective = True
                        elif self.selective_model is not None and main_confidence < 0.15:
                            # También usar modelo selectivo si la confianza es muy baja
                            logger.info(f"🎯 Confianza baja ({main_confidence:.4f}), probando modelo selectivo...")
                            use_selective = True
                        
                        if use_selective:
                            # Usar modelo selectivo
                            selective_output = self.selective_model(input_tensor)
                            selective_probs = F.softmax(selective_output / self.breed_temperature, dim=1)
                            selective_confidence, selective_pred = torch.max(selective_probs, 1)
                            selective_breed = self.selective_idx_to_breed[selective_pred.item()]
                            selective_conf = float(selective_confidence.item())
                            
                            # Decidir cuál resultado usar
                            if selective_conf > main_confidence * 1.2:  # 20% ventaja al modelo selectivo
                                logger.info(f"🎯 Usando modelo selectivo: {selective_breed} (conf: {selective_conf:.4f})")
                                results['breed'] = selective_breed
                                results['breed_confidence'] = selective_conf
                                results['model_used'] = 'selective'
                                
                                # Top-3 del modelo selectivo
                                top3_values, top3_indices = torch.topk(selective_probs, min(3, len(self.selective_classes)), dim=1)
                                results['breed_top3'] = [
                                    {
                                        'breed': self.selective_idx_to_breed[idx.item()],
                                        'confidence': float(prob.item())
                                    }
                                    for prob, idx in zip(top3_values[0], top3_indices[0])
                                ]
                            else:
                                logger.info(f"🎯 Modelo principal mejor: {main_breed} (conf: {main_confidence:.4f})")
                                results['breed'] = main_breed
                                results['breed_confidence'] = main_confidence
                                results['model_used'] = 'main'
                                
                                # Top-3 del modelo principal
                                top3_values, top3_indices = torch.topk(breed_probs, 3, dim=1)
                                results['breed_top3'] = [
                                    {
                                        'breed': self.breed_classes[idx.item()],
                                        'confidence': float(prob.item())
                                    }
                                    for prob, idx in zip(top3_values[0], top3_indices[0])
                                ]
                        else:
                            # Usar solo modelo principal
                            results['breed'] = main_breed
                            results['breed_confidence'] = main_confidence
                            results['model_used'] = 'main'
                            
                            # Top-3 predicciones del modelo principal
                            top3_values, top3_indices = torch.topk(breed_probs, 3, dim=1)
                            results['breed_top3'] = [
                                {
                                    'breed': self.breed_classes[idx.item()],
                                    'confidence': float(prob.item())
                                }
                                for prob, idx in zip(top3_values[0], top3_indices[0])
                            ]
                        
                        logger.info(f"🐕 Raza: {results['breed']} "
                                  f"(confianza: {results['breed_confidence']:.4f}) "
                                  f"[{results.get('model_used', 'main')}]")
                else:
                    logger.error("❌ Modelo de razas es None o no hay clases!")
                    logger.error(f"❌ breed_model: {self.breed_model is not None}, breed_classes: {len(self.breed_classes) if self.breed_classes else 0}")
                    results['error'] = "Modelo de razas no disponible"
            elif results['is_dog'] and results['binary_confidence'] < confidence_threshold:
                logger.info(f"🐕 Perro detectado pero con baja confianza ({results['binary_confidence']:.4f} < {confidence_threshold})")
                # Si es perro pero con baja confianza, intentar predicción de raza de todos modos
                if self.breed_model is not None and self.breed_classes:
                    logger.info("🐕 Intentando clasificación de raza con baja confianza...")
                    with torch.no_grad():
                        breed_output = self.breed_model(input_tensor)
                        # Aplicar temperature scaling también para baja confianza
                        breed_probs = F.softmax(breed_output / self.breed_temperature, dim=1)
                        
                        # Top-1 predicción
                        breed_confidence, breed_pred = torch.max(breed_probs, 1)
                        results['breed'] = f"Posiblemente: {self.breed_classes[breed_pred.item()]}"
                        results['breed_confidence'] = float(breed_confidence.item())
                        
                        # Top-3 predicciones
                        top3_values, top3_indices = torch.topk(breed_probs, 3, dim=1)
                        results['breed_top3'] = [
                            {
                                'breed': self.breed_classes[idx.item()],
                                'confidence': float(prob.item())
                            }
                            for prob, idx in zip(top3_values[0], top3_indices[0])
                        ]
                        
                        logger.info(f"🐕 Raza (baja confianza): {results['breed']} "
                                  f"(confianza: {results['breed_confidence']:.4f})")
                else:
                    results['breed'] = "Confianza insuficiente para determinar raza"
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return {
                'error': f"Error procesando imagen: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self):
        """Información sobre los modelos cargados"""
        return {
            'binary_model_loaded': self.binary_model is not None,
            'breed_model_loaded': self.breed_model is not None,
            'selective_model_loaded': self.selective_model is not None,
            'binary_architecture': 'ResNet18',
            'breed_architecture': 'ResNet50 (Balanceado)',
            'selective_architecture': 'ResNet34 (6 breeds)',
            'num_breeds': len(self.breed_classes),
            'num_selective_breeds': len(self.selective_classes),
            'device': str(self.device),
            'breed_classes': self.breed_classes[:10],  # Solo primeras 10 para no sobrecargar
            'selective_breeds': list(self.selective_classes.keys()) if self.selective_classes else [],
            'breed_temperature': self.breed_temperature,
            'binary_temperature': self.binary_temperature,
            'dataset_balanced': True,
            'images_per_class': 161
        }
    
    def adjust_temperature(self, breed_temp=None, binary_temp=None):
        """Ajustar temperaturas dinámicamente"""
        if breed_temp is not None:
            self.breed_temperature = breed_temp
            logger.info(f"🌡️ Temperature para razas ajustada a: {breed_temp}")
        if binary_temp is not None:
            self.binary_temperature = binary_temp 
            logger.info(f"🌡️ Temperature para binario ajustada a: {binary_temp}")

# =============================================================================
# APLICACIÓN FLASK CON FRONTEND
# =============================================================================

app = Flask(__name__)
classifier = HierarchicalDogClassifier()

# HTML Template para el frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐕 Clasificador Jerárquico de Perros</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #ff6b6b, #ffa500);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #ff6b6b;
            background: #fafafa;
        }
        
        .upload-area.dragover {
            border-color: #ff6b6b;
            background: #fff5f5;
        }
        
        #fileInput {
            display: none;
        }
        
        .upload-icon {
            font-size: 4em;
            color: #ddd;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.3em;
            color: #666;
            margin-bottom: 15px;
        }
        
        .upload-subtext {
            color: #999;
            font-size: 0.9em;
        }
        
        .btn {
            background: linear-gradient(135deg, #ff6b6b, #ffa500);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .preview-container {
            display: none;
            margin-bottom: 30px;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .results {
            display: none;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
        }
        
        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid #eee;
        }
        
        .result-item:last-child {
            border-bottom: none;
        }
        
        .result-label {
            font-weight: bold;
            color: #333;
        }
        
        .result-value {
            color: #666;
        }
        
        .confidence-bar {
            width: 200px;
            height: 10px;
            background: #eee;
            border-radius: 5px;
            overflow: hidden;
            margin-left: 15px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #ffa500);
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        
        .breed-list {
            margin-top: 15px;
        }
        
        .breed-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .breed-item:last-child {
            border-bottom: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #ff6b6b;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid #c62828;
        }
        
        .success {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid #2e7d32;
        }
        
        @media (max-width: 600px) {
            .container {
                margin: 10px;
                border-radius: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .content {
                padding: 20px;
            }
            
            .confidence-bar {
                width: 100px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐕 Clasificador de Perros</h1>
            <p>Sistema Jerárquico con IA • ResNet18 + ResNet34</p>
        </div>
        
        <div class="content">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Haz clic aquí o arrastra una imagen</div>
                <div class="upload-subtext">Formatos: JPG, PNG, GIF • Máximo 10MB</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="preview-container">
                <img class="preview-image" id="previewImage" alt="Vista previa">
            </div>
            
            <div style="text-align: center;">
                <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
                    🔍 Analizar Imagen
                </button>
                
                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px;">
                    <label for="tempSlider" style="display: block; margin-bottom: 10px; font-weight: bold;">
                        🌡️ Temperatura de Calibración: <span id="tempValue">10.0</span>
                    </label>
                    <input type="range" id="tempSlider" min="1" max="15" step="0.5" value="10.0" 
                           style="width: 100%; margin-bottom: 10px;" onchange="updateTemperature()">
                    <div style="font-size: 0.9em; color: #666;">
                        Menor = Más extremo | Mayor = Más balanceado
                    </div>
                </div>
            </div>
            
            <div class="loading">
                <div class="spinner"></div>
                <div>Analizando imagen con IA...</div>
            </div>
            
            <div class="results" id="results">
                <!-- Resultados se mostrarán aquí -->
            </div>
        </div>
    </div>

    <script>
        let selectedImage = null;
        
        // Configurar eventos de drag & drop
        const uploadArea = document.querySelector('.upload-area');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.querySelector('.preview-container');
        const previewImage = document.getElementById('previewImage');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultsDiv = document.getElementById('results');
        const loadingDiv = document.querySelector('.loading');
        
        // Eventos drag & drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Por favor selecciona un archivo de imagen válido.');
                return;
            }
            
            if (file.size > 10 * 1024 * 1024) {
                showError('El archivo es demasiado grande. Máximo 10MB.');
                return;
            }
            
            selectedImage = file;
            
            // Mostrar preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.style.display = 'block';
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);
            
            // Ocultar resultados anteriores
            resultsDiv.style.display = 'none';
        }
        
        async function analyzeImage() {
            if (!selectedImage) return;
            
            // Mostrar loading
            loadingDiv.style.display = 'block';
            resultsDiv.style.display = 'none';
            analyzeBtn.disabled = true;
            
            try {
                const formData = new FormData();
                formData.append('image', selectedImage);
                
                const response = await fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResults(data);
                }
                
            } catch (error) {
                showError('Error conectando con el servidor: ' + error.message);
            } finally {
                loadingDiv.style.display = 'none';
                analyzeBtn.disabled = false;
            }
        }
        
        function showResults(data) {
            let html = '<h3>📊 Resultados del Análisis</h3>';
            
            // Resultado binario
            html += `
                <div class="result-item">
                    <div class="result-label">🔍 Detección:</div>
                    <div style="display: flex; align-items: center;">
                        <span class="result-value">${data.is_dog ? '🐕 ES UN PERRO' : '❌ NO ES UN PERRO'}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${data.binary_confidence * 100}%"></div>
                        </div>
                        <span style="margin-left: 10px; color: #666;">${(data.binary_confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
            
            // Resultado de raza si es perro
            if (data.is_dog && data.breed) {
                if (data.breed_confidence > 0) {
                    html += `
                        <div class="result-item">
                            <div class="result-label">🏷️ Raza Principal:</div>
                            <div style="display: flex; align-items: center;">
                                <span class="result-value">${data.breed}</span>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${data.breed_confidence * 100}%"></div>
                                </div>
                                <span style="margin-left: 10px; color: #666;">${(data.breed_confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="result-item">
                            <div class="result-label">🏷️ Raza:</div>
                            <span class="result-value">${data.breed}</span>
                        </div>
                    `;
                }
                
                // Top 3 razas
                if (data.breed_top3 && data.breed_top3.length > 0) {
                    html += `
                        <div class="result-item">
                            <div class="result-label">🥇 Top 3 Razas:</div>
                            <div class="breed-list">
                    `;
                    
                    data.breed_top3.forEach((breed, index) => {
                        const medal = ['🥇', '🥈', '🥉'][index];
                        html += `
                            <div class="breed-item">
                                <span>${medal} ${breed.breed}</span>
                                <div style="display: flex; align-items: center;">
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${breed.confidence * 100}%"></div>
                                    </div>
                                    <span style="margin-left: 10px; color: #666;">${(breed.confidence * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        `;
                    });
                    
                    html += `
                            </div>
                        </div>
                    `;
                }
            }
            
            // Información técnica
            html += `
                <div class="result-item">
                    <div class="result-label">⚙️ Modelos:</div>
                    <span class="result-value">ResNet18 (Binario) + ResNet34 (Razas)</span>
                </div>
                <div class="result-item">
                    <div class="result-label">🌡️ Temperatura:</div>
                    <span class="result-value">${data.temperature || 'N/A'}</span>
                </div>
                <div class="result-item">
                    <div class="result-label">⏰ Procesado:</div>
                    <span class="result-value">${new Date(data.timestamp).toLocaleTimeString()}</span>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
        
        function showError(message) {
            resultsDiv.innerHTML = `<div class="error">❌ ${message}</div>`;
            resultsDiv.style.display = 'block';
        }
        
        function updateTemperature() {
            const slider = document.getElementById('tempSlider');
            const tempValue = document.getElementById('tempValue');
            tempValue.textContent = slider.value;
            
            // Enviar nueva temperatura al servidor
            fetch('http://localhost:5000/adjust_temp', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    breed_temperature: parseFloat(slider.value)
                })
            }).then(response => response.json())
              .then(data => {
                  if (data.error) {
                      console.error('Error ajustando temperatura:', data.error);
                  } else {
                      console.log('✅ Temperatura ajustada:', data.breed_temperature);
                  }
              }).catch(error => {
                  console.error('Error:', error);
              });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Página principal con frontend"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/test')
def test():
    """Endpoint de prueba"""
    logger.info("🧪 Endpoint de prueba llamado")
    return jsonify({
        'status': 'ok',
        'message': 'Servidor funcionando correctamente',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint para predicción"""
    logger.info("🚀 Petición de predicción recibida")
    try:
        logger.info("🔍 Verificando archivos en la petición...")
        if 'image' not in request.files:
            logger.error("❌ No hay imagen en la petición")
            return jsonify({'error': 'No se encontró imagen en la petición'})
        
        file = request.files['image']
        logger.info(f"📁 Archivo recibido: {file.filename}")
        if file.filename == '':
            logger.error("❌ Nombre de archivo vacío")
            return jsonify({'error': 'No se seleccionó archivo'})
        
        logger.info("🖼️ Procesando imagen...")
        # Convertir a PIL Image
        image = Image.open(io.BytesIO(file.read()))
        logger.info(f"✅ Imagen cargada: {image.size}")
        
        # Hacer predicción con umbral más bajo para detectar más razas
        logger.info("🤖 Iniciando predicción...")
        result = classifier.predict_image(image, confidence_threshold=0.35)
        logger.info(f"📊 Resultado: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error en API: {e}")
        return jsonify({'error': f'Error procesando imagen: {str(e)}'})

@app.route('/info')
def model_info():
    """Información sobre los modelos"""
    return jsonify(classifier.get_model_info())

@app.route('/health')
def health_check():
    """Health check del servicio"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'binary': classifier.binary_model is not None,
            'breed': classifier.breed_model is not None
        }
    })

@app.route('/adjust_temp', methods=['POST'])
def adjust_temperature():
    """Ajustar temperatura de calibración"""
    try:
        data = request.get_json()
        breed_temp = data.get('breed_temperature')
        binary_temp = data.get('binary_temperature')
        
        classifier.adjust_temperature(breed_temp, binary_temp)
        
        return jsonify({
            'status': 'success',
            'message': 'Temperaturas ajustadas',
            'breed_temperature': classifier.breed_temperature,
            'binary_temperature': classifier.binary_temperature
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal"""
    print("🔥" * 80)
    print("🐕 CLASIFICADOR JERÁRQUICO DE PERROS - SISTEMA INTEGRADO")
    print("🔥" * 80)
    print("📊 Modelos:")
    print("   🔸 Binario: ResNet18 (perro/no perro)")
    print("   🔸 Razas: ResNet50 BALANCEADO (50 razas)")
    print("   🔸 Selectivo: ResNet34 (6 razas problemáticas)")
    print("🔥" * 80)
    
    # Mostrar información de modelos
    info = classifier.get_model_info()
    print("📋 Estado de modelos:")
    print(f"   Binary cargado: {'✅' if info['binary_model_loaded'] else '❌'}")
    print(f"   Breeds cargado: {'✅' if info['breed_model_loaded'] else '❌'}")
    print(f"   Selectivo cargado: {'✅' if info['selective_model_loaded'] else '❌'}")
    print(f"   Dispositivo: {info['device']}")
    print(f"   Razas disponibles: {info['num_breeds']}")
    print(f"   Dataset balanceado: ✅ ({info['images_per_class']} img/raza)")
    if info['selective_model_loaded']:
        print(f"   Razas selectivas: {info['num_selective_breeds']} ({', '.join(info['selective_breeds'])})")
    
    if not info['binary_model_loaded'] or not info['breed_model_loaded']:
        print("\n⚠️ ADVERTENCIA: Algunos modelos no están cargados")
        print("   Verifica que existan los archivos:")
        print("   - binary_models/best_fast_binary_model.pth")
        print("   - autonomous_breed_models/best_breed_model.pth")
    
    print("\n🚀 Iniciando servidor web...")
    print("📱 Abre tu navegador en: http://localhost:5000")
    print("🔥" * 80)
    
    # Iniciar servidor Flask con CORS habilitado
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == "__main__":
    main()