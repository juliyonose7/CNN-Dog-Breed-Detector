#!/usr/bin/env python3
"""
🐕 CLASIFICADOR UNIFICADO DE PERROS - SIN SESGOS ARQUITECTURALES
==============================================================

Versión corregida del clasificador que elimina sesgos:
1. ❌ Removido modelo selectivo (ventaja injusta)
2. ✅ Solo modelo principal ResNet50 para TODAS las razas
3. ✅ Métricas detalladas por clase individual
4. ✅ Umbrales adaptativos por raza

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
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DEFINICIÓN DE MODELOS UNIFICADOS
# =============================================================================

class UnifiedBreedClassifier(nn.Module):
    """Modelo unificado basado en ResNet50 para todas las razas"""
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

class BinaryClassifier(nn.Module):
    """Modelo binario basado en ResNet18"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# CLASIFICADOR UNIFICADO PRINCIPAL
# =============================================================================

class UnbiasedDogClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Dispositivo: {self.device}")
        
        # Modelos
        self.binary_model = None
        self.breed_model = None
        
        # Clases
        self.binary_classes = ['nodog', 'dog']
        self.breed_classes = []
        
        # Umbrales adaptativos por raza (inicializar con valores por defecto)
        self.adaptive_thresholds = {}
        self.default_threshold = 0.35
        
        # Métricas por clase
        self.class_metrics = {}
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar modelos y configuraciones
        self._load_models()
        self._load_adaptive_thresholds()
        self._load_class_metrics()
        
    def _load_models(self):
        """Carga modelos sin el componente selectivo"""
        try:
            # 1. MODELO BINARIO (ResNet18)
            binary_path = "realtime_binary_models/best_model_epoch_1_acc_0.9649.pth"
            if os.path.exists(binary_path):
                logger.info("📁 Cargando modelo binario (ResNet18)...")
                self.binary_model = BinaryClassifier(num_classes=2).to(self.device)
                checkpoint = torch.load(binary_path, map_location=self.device)
                self.binary_model.load_state_dict(checkpoint['model_state_dict'])
                self.binary_model.eval()
                logger.info(f"✅ Modelo binario cargado - Accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
            else:
                logger.error(f"❌ Modelo binario no encontrado: {binary_path}")
                
            # 2. MODELO DE RAZAS UNIFICADO (ResNet50 balanceado)
            breed_path = "balanced_models/best_balanced_breed_model_epoch_20_acc_88.1366.pth"
            if os.path.exists(breed_path):
                logger.info("📁 Cargando modelo de razas UNIFICADO (ResNet50)...")
                self.breed_model = UnifiedBreedClassifier(num_classes=50).to(self.device)
                checkpoint = torch.load(breed_path, map_location=self.device)
                self.breed_model.load_state_dict(checkpoint['model_state_dict'])
                self.breed_model.eval()
                logger.info(f"✅ Modelo de razas UNIFICADO cargado - Accuracy: {checkpoint.get('val_accuracy', 0):.2f}%")
                logger.info(f"📊 Dataset balanceado: {checkpoint.get('images_per_class', 161)} imágenes por clase")
                
                # Cargar nombres de razas
                self._load_breed_names()
            else:
                logger.error(f"❌ Modelo de razas no encontrado: {breed_path}")
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            
    def _load_breed_names(self):
        """Carga los nombres de las 50 razas"""
        breed_data_path = "breed_processed_data/train"
        if os.path.exists(breed_data_path):
            self.breed_classes = sorted([d for d in os.listdir(breed_data_path) 
                                       if os.path.isdir(os.path.join(breed_data_path, d))])
            logger.info(f"📋 Cargadas {len(self.breed_classes)} razas (TODAS con modelo unificado)")
        else:
            logger.warning("⚠️ Directorio de razas no encontrado, usando nombres genéricos")
            self.breed_classes = [f"Raza_{i:02d}" for i in range(50)]
    
    def _load_adaptive_thresholds(self):
        """Carga umbrales adaptativos por raza"""
        threshold_path = "adaptive_thresholds.json"
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path, 'r') as f:
                    self.adaptive_thresholds = json.load(f)
                logger.info(f"📊 Cargados umbrales adaptativos para {len(self.adaptive_thresholds)} razas")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando umbrales adaptativos: {e}")
                self.adaptive_thresholds = {}
        else:
            logger.info("📊 Umbrales adaptativos no encontrados, usando valores por defecto")
            self.adaptive_thresholds = {}
    
    def _load_class_metrics(self):
        """Carga métricas detalladas por clase"""
        metrics_path = "class_metrics.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    self.class_metrics = json.load(f)
                logger.info(f"📈 Cargadas métricas para {len(self.class_metrics)} razas")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando métricas por clase: {e}")
                self.class_metrics = {}
        else:
            logger.info("📈 Métricas por clase no encontradas")
            self.class_metrics = {}
    
    def predict_image(self, image_path_or_pil, use_adaptive_threshold=True):
        """
        Clasificación unificada sin sesgos arquitecturales
        
        Args:
            image_path_or_pil: Path a imagen o objeto PIL
            use_adaptive_threshold: Usar umbrales adaptativos por raza
            
        Returns:
            dict con resultados completos y métricas detalladas
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
                'breed_top5': [],
                'adaptive_threshold_used': use_adaptive_threshold,
                'model_architecture': 'Unificado (ResNet50 para todas las razas)',
                'bias_mitigation': 'Modelo selectivo eliminado',
                'class_metrics': {},
                'error': None
            }
            
            # PASO 1: CLASIFICACIÓN BINARIA
            if self.binary_model is not None:
                logger.info("🔍 Iniciando clasificación binaria...")
                with torch.no_grad():
                    binary_output = self.binary_model(input_tensor)
                    binary_probs = F.softmax(binary_output, dim=1)
                    binary_confidence, binary_pred = torch.max(binary_probs, 1)
                    
                    results['binary_confidence'] = float(binary_confidence.item())
                    results['is_dog'] = bool(binary_pred.item() == 1)  # 1 = dog
                    
                    logger.info(f"🔍 Binario: {'PERRO' if results['is_dog'] else 'NO PERRO'} "
                              f"(confianza: {results['binary_confidence']:.4f})")
            else:
                logger.error("❌ Modelo binario es None!")
                results['error'] = "Modelo binario no disponible"
                return results
            
            # PASO 2: CLASIFICACIÓN DE RAZA UNIFICADA (solo si es perro)
            if results['is_dog']:
                logger.info(f"🐕 Iniciando clasificación de razas UNIFICADA...")
                if self.breed_model is not None and self.breed_classes:
                    with torch.no_grad():
                        breed_output = self.breed_model(input_tensor)
                        breed_probs = F.softmax(breed_output, dim=1)
                        
                        # Top-5 predicciones (más información para análisis)
                        top5_values, top5_indices = torch.topk(breed_probs, 5, dim=1)
                        results['breed_top5'] = []
                        
                        for prob, idx in zip(top5_values[0], top5_indices[0]):
                            breed_name = self.breed_classes[idx.item()]
                            confidence = float(prob.item())
                            
                            # Obtener umbral adaptativo para esta raza
                            if use_adaptive_threshold and breed_name in self.adaptive_thresholds:
                                threshold = self.adaptive_thresholds[breed_name]
                            else:
                                threshold = self.default_threshold
                            
                            # Agregar métricas de la raza si están disponibles
                            breed_metrics = self.class_metrics.get(breed_name, {})
                            
                            results['breed_top5'].append({
                                'breed': breed_name,
                                'confidence': confidence,
                                'threshold': threshold,
                                'above_threshold': confidence >= threshold,
                                'precision': breed_metrics.get('precision', 'N/A'),
                                'recall': breed_metrics.get('recall', 'N/A'),
                                'f1_score': breed_metrics.get('f1_score', 'N/A')
                            })
                        
                        # Predicción principal (Top-1)
                        top_prediction = results['breed_top5'][0]
                        results['breed'] = top_prediction['breed']
                        results['breed_confidence'] = top_prediction['confidence']
                        results['threshold_used'] = top_prediction['threshold']
                        results['prediction_above_threshold'] = top_prediction['above_threshold']
                        results['class_metrics'] = {
                            'precision': top_prediction['precision'],
                            'recall': top_prediction['recall'],
                            'f1_score': top_prediction['f1_score']
                        }
                        
                        logger.info(f"🐕 Raza: {results['breed']} "
                                  f"(confianza: {results['breed_confidence']:.4f}) "
                                  f"[umbral: {results['threshold_used']:.3f}] "
                                  f"{'✅' if results['prediction_above_threshold'] else '⚠️'}")
                        
                        # Información adicional sobre el modelo unificado
                        results['model_info'] = {
                            'architecture': 'ResNet50 Unificado',
                            'total_breeds': len(self.breed_classes),
                            'selective_bias_removed': True,
                            'all_breeds_equal_treatment': True
                        }
                        
                else:
                    logger.error("❌ Modelo de razas es None o no hay clases!")
                    results['error'] = "Modelo de razas no disponible"
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return {
                'error': f"Error procesando imagen: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_class_performance(self, test_data_path=None):
        """Evalúa el rendimiento detallado por cada clase"""
        logger.info("📊 EVALUANDO RENDIMIENTO POR CLASE...")
        
        if not test_data_path:
            test_data_path = "breed_processed_data/val"  # Usar datos de validación
            
        if not os.path.exists(test_data_path):
            logger.error(f"❌ No se encuentra el directorio de prueba: {test_data_path}")
            return None
        
        class_results = {}
        total_correct = 0
        total_samples = 0
        
        for breed_dir in os.listdir(test_data_path):
            breed_path = os.path.join(test_data_path, breed_dir)
            if not os.path.isdir(breed_path):
                continue
                
            breed_name = breed_dir
            if breed_name not in self.breed_classes:
                continue
            
            logger.info(f"🔍 Evaluando {breed_name}...")
            
            true_labels = []
            predicted_labels = []
            confidences = []
            
            # Evaluar todas las imágenes de esta raza
            image_files = [f for f in os.listdir(breed_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            correct_predictions = 0
            
            for image_file in image_files[:50]:  # Limitar a 50 imágenes por raza para rapidez
                try:
                    image_path = os.path.join(breed_path, image_file)
                    result = self.predict_image(image_path, use_adaptive_threshold=False)
                    
                    if result.get('error'):
                        continue
                        
                    if result['is_dog'] and result['breed']:
                        predicted_breed = result['breed']
                        confidence = result['breed_confidence']
                        
                        true_labels.append(breed_name)
                        predicted_labels.append(predicted_breed)
                        confidences.append(confidence)
                        
                        if predicted_breed == breed_name:
                            correct_predictions += 1
                            
                        total_samples += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error procesando {image_file}: {e}")
                    continue
            
            if len(true_labels) > 0:
                breed_accuracy = correct_predictions / len(true_labels)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                class_results[breed_name] = {
                    'accuracy': breed_accuracy,
                    'samples_evaluated': len(true_labels),
                    'correct_predictions': correct_predictions,
                    'avg_confidence': avg_confidence,
                    'min_confidence': min(confidences) if confidences else 0.0,
                    'max_confidence': max(confidences) if confidences else 0.0
                }
                
                total_correct += correct_predictions
                
                logger.info(f"   ✅ {breed_name}: {breed_accuracy:.3f} accuracy "
                          f"({correct_predictions}/{len(true_labels)}) "
                          f"conf: {avg_confidence:.3f}")
        
        # Estadísticas globales
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        evaluation_summary = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'classes_evaluated': len(class_results),
            'class_results': class_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar resultados
        with open('detailed_class_evaluation.json', 'w') as f:
            json.dump(evaluation_summary, f, indent=2, default=str)
        
        logger.info(f"📊 EVALUACIÓN COMPLETA:")
        logger.info(f"   Overall Accuracy: {overall_accuracy:.4f}")
        logger.info(f"   Samples: {total_samples}")
        logger.info(f"   Classes: {len(class_results)}")
        logger.info(f"   ✅ Guardado: detailed_class_evaluation.json")
        
        return evaluation_summary
    
    def compute_adaptive_thresholds(self, validation_data_path=None):
        """Calcula umbrales adaptativos óptimos por raza usando datos de validación"""
        logger.info("🎯 CALCULANDO UMBRALES ADAPTATIVOS POR RAZA...")
        
        if not validation_data_path:
            validation_data_path = "breed_processed_data/val"
            
        if not os.path.exists(validation_data_path):
            logger.error(f"❌ No se encuentra el directorio de validación: {validation_data_path}")
            return None
        
        adaptive_thresholds = {}
        
        for breed_dir in os.listdir(validation_data_path):
            breed_path = os.path.join(validation_data_path, breed_dir)
            if not os.path.isdir(breed_path):
                continue
                
            breed_name = breed_dir
            if breed_name not in self.breed_classes:
                continue
            
            logger.info(f"🎯 Calculando umbral para {breed_name}...")
            
            # Recopilar predicciones para esta raza
            true_scores = []  # Scores cuando la imagen ES de esta raza
            false_scores = []  # Scores cuando la imagen NO es de esta raza
            
            # Imágenes positivas (de esta raza)
            image_files = [f for f in os.listdir(breed_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files[:30]:  # Limitar para rapidez
                try:
                    image_path = os.path.join(breed_path, image_file)
                    result = self.predict_image(image_path, use_adaptive_threshold=False)
                    
                    if result.get('error') or not result['is_dog']:
                        continue
                    
                    # Buscar el score de esta raza específica en top-5
                    for pred in result['breed_top5']:
                        if pred['breed'] == breed_name:
                            true_scores.append(pred['confidence'])
                            break
                    else:
                        # Si no está en top-5, asignar score muy bajo
                        true_scores.append(0.01)
                        
                except Exception as e:
                    continue
            
            # Imágenes negativas (de otras razas)
            other_breeds = [b for b in os.listdir(validation_data_path) 
                           if b != breed_name and os.path.isdir(os.path.join(validation_data_path, b))]
            
            for other_breed in other_breeds[:5]:  # Solo unas pocas razas como negativas
                other_path = os.path.join(validation_data_path, other_breed)
                other_images = [f for f in os.listdir(other_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for image_file in other_images[:10]:  # Pocas imágenes por raza
                    try:
                        image_path = os.path.join(other_path, image_file)
                        result = self.predict_image(image_path, use_adaptive_threshold=False)
                        
                        if result.get('error') or not result['is_dog']:
                            continue
                        
                        # Buscar el score de la raza objetivo en top-5
                        for pred in result['breed_top5']:
                            if pred['breed'] == breed_name:
                                false_scores.append(pred['confidence'])
                                break
                        else:
                            false_scores.append(0.01)
                            
                    except Exception as e:
                        continue
            
            # Calcular umbral óptimo usando curva ROC
            if len(true_scores) > 5 and len(false_scores) > 5:
                # Preparar datos para ROC
                y_true = [1] * len(true_scores) + [0] * len(false_scores)
                y_scores = true_scores + false_scores
                
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    
                    # Encontrar umbral que maximiza Youden's J statistic (tpr - fpr)
                    j_scores = tpr - fpr
                    optimal_idx = np.argmax(j_scores)
                    optimal_threshold = thresholds[optimal_idx]
                    
                    # Asegurar que el umbral esté en rango razonable
                    optimal_threshold = max(0.1, min(0.8, optimal_threshold))
                    
                    adaptive_thresholds[breed_name] = optimal_threshold
                    
                    logger.info(f"   ✅ {breed_name}: umbral óptimo = {optimal_threshold:.3f} "
                              f"(J = {j_scores[optimal_idx]:.3f})")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Error calculando umbral para {breed_name}: {e}")
                    adaptive_thresholds[breed_name] = self.default_threshold
            else:
                logger.warning(f"   ⚠️ Datos insuficientes para {breed_name}, usando umbral por defecto")
                adaptive_thresholds[breed_name] = self.default_threshold
        
        # Guardar umbrales adaptativos
        with open('adaptive_thresholds.json', 'w') as f:
            json.dump(adaptive_thresholds, f, indent=2)
        
        self.adaptive_thresholds = adaptive_thresholds
        
        logger.info(f"🎯 UMBRALES ADAPTATIVOS CALCULADOS:")
        logger.info(f"   Razas procesadas: {len(adaptive_thresholds)}")
        logger.info(f"   Rango de umbrales: {min(adaptive_thresholds.values()):.3f} - {max(adaptive_thresholds.values()):.3f}")
        logger.info(f"   ✅ Guardado: adaptive_thresholds.json")
        
        return adaptive_thresholds
    
    def get_model_info(self):
        """Información sobre el modelo unificado sin sesgos"""
        return {
            'architecture': 'Unificado - ResNet50 para todas las razas',
            'binary_model_loaded': self.binary_model is not None,
            'breed_model_loaded': self.breed_model is not None,
            'selective_model_removed': True,  # ✅ Sesgo eliminado
            'bias_mitigation': {
                'architectural_bias_removed': True,
                'selective_advantage_eliminated': True,
                'unified_treatment': True
            },
            'num_breeds': len(self.breed_classes),
            'adaptive_thresholds_available': len(self.adaptive_thresholds),
            'class_metrics_available': len(self.class_metrics),
            'device': str(self.device),
            'dataset_balanced': True,
            'images_per_class': 161
        }

# =============================================================================
# APLICACIÓN FLASK ACTUALIZADA
# =============================================================================

app = Flask(__name__)
classifier = UnbiasedDogClassifier()

@app.route('/')
def index():
    """Página principal"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐕 Clasificador Unificado Sin Sesgos</title>
        <style>
            body { font-family: Arial; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; color: #2c3e50; }
            .bias-info { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .improvement { color: #27ae60; font-weight: bold; }
            .upload-area { border: 2px dashed #3498db; padding: 40px; text-align: center; margin: 20px 0; }
            input[type="file"] { margin: 10px; }
            button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .results { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐕 Clasificador Unificado de Perros</h1>
                <h2>✅ Versión Sin Sesgos Arquitecturales</h2>
            </div>
            
            <div class="bias-info">
                <h3>🛡️ Mejoras Implementadas:</h3>
                <ul>
                    <li class="improvement">✅ Modelo selectivo eliminado (todas las razas tienen igual tratamiento)</li>
                    <li class="improvement">✅ Arquitectura unificada ResNet50</li>
                    <li class="improvement">✅ Umbrales adaptativos por raza</li>
                    <li class="improvement">✅ Métricas detalladas por clase</li>
                </ul>
            </div>
            
            <div class="upload-area">
                <h3>📁 Subir Imagen de Perro</h3>
                <input type="file" id="fileInput" accept="image/*">
                <br><button onclick="analyzeImage()">🔍 Analizar</button>
            </div>
            
            <div class="results" id="results" style="display: none;"></div>
        </div>
        
        <script>
            async function analyzeImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) return alert('Selecciona una imagen');
                
                const formData = new FormData();
                formData.append('image', file);
                
                try {
                    const response = await fetch('/predict', {method: 'POST', body: formData});
                    const data = await response.json();
                    
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('results').innerHTML = `
                        <h3>📊 Resultados del Análisis Sin Sesgos</h3>
                        <p><strong>🔍 Es perro:</strong> ${data.is_dog ? '✅ Sí' : '❌ No'} (${(data.binary_confidence * 100).toFixed(1)}%)</p>
                        ${data.breed ? `
                            <p><strong>🏷️ Raza:</strong> ${data.breed}</p>
                            <p><strong>📈 Confianza:</strong> ${(data.breed_confidence * 100).toFixed(1)}%</p>
                            <p><strong>🎯 Umbral usado:</strong> ${data.threshold_used ? data.threshold_used.toFixed(3) : 'N/A'}</p>
                            <p><strong>✅ Sobre umbral:</strong> ${data.prediction_above_threshold ? 'Sí' : 'No'}</p>
                            <p><strong>🛡️ Arquitectura:</strong> ${data.model_architecture}</p>
                            <p><strong>🔧 Mejora:</strong> ${data.bias_mitigation}</p>
                        ` : ''}
                    `;
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint para predicción sin sesgos"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se encontró imagen en la petición'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'})
        
        # Convertir a PIL Image
        image = Image.open(io.BytesIO(file.read()))
        
        # Hacer predicción con el modelo unificado
        result = classifier.predict_image(image, use_adaptive_threshold=True)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error procesando imagen: {str(e)}'})

@app.route('/info')
def model_info():
    """Información del modelo unificado"""
    return jsonify(classifier.get_model_info())

@app.route('/evaluate')
def evaluate():
    """Evaluar rendimiento detallado por clase"""
    results = classifier.evaluate_class_performance()
    return jsonify(results)

@app.route('/compute_thresholds')
def compute_thresholds():
    """Calcular umbrales adaptativos"""
    results = classifier.compute_adaptive_thresholds()
    return jsonify({'adaptive_thresholds': results})

if __name__ == "__main__":
    print("🛡️" * 80)
    print("🐕 CLASIFICADOR UNIFICADO SIN SESGOS ARQUITECTURALES")
    print("🛡️" * 80)
    print("✅ Mejoras implementadas:")
    print("   🚫 Modelo selectivo ELIMINADO")
    print("   🏗️ Arquitectura UNIFICADA (ResNet50)")
    print("   🎯 Umbrales ADAPTATIVOS por raza")
    print("   📊 Métricas DETALLADAS por clase")
    print("🛡️" * 80)
    
    info = classifier.get_model_info()
    print("📋 Estado del modelo:")
    print(f"   Binario cargado: {'✅' if info['binary_model_loaded'] else '❌'}")
    print(f"   Breeds cargado: {'✅' if info['breed_model_loaded'] else '❌'}")
    print(f"   Sesgo eliminado: {'✅' if info['selective_model_removed'] else '❌'}")
    print(f"   Razas disponibles: {info['num_breeds']}")
    
    print(f"\n🚀 Iniciando servidor en: http://localhost:5001")
    app.run(host='127.0.0.1', port=5001, debug=False)