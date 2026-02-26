#!/usr/bin/env python3
"""
Unbiased Unified Dog Breed Classifier API.

This module provides a Flask-based REST API for dog breed classification
that eliminates architectural biases present in selective model approaches.

Key Design Decisions:
    1. Removed selective model component (unfair advantage to certain breeds)
    2. Single unified ResNet50 model for all breeds (equal treatment)
    3. Robust binary classification (dog/not-dog) with ResNet18
    4. Adaptive confidence thresholds per breed

Architecture:
    - Binary Classifier: ResNet18 for dog detection
    - Breed Classifier: Unified ResNet50 for all 50 breeds
    - Adaptive Thresholds: Per-breed confidence calibration

API Endpoints:
    - POST /predict: Unified breed classification
    - GET /model-info: Architecture and configuration details
    - GET /health: API health status
    - GET /adaptive-thresholds: Current threshold configuration

Author: AI System
Date: 2024
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

class UnifiedBreedClassifier(nn.Module):
    """
    Unified breed classification model based on ResNet50.
    
    A single model for all breeds, ensuring equal treatment without
    architectural bias favoring specific breeds.
    
    Attributes:
        backbone (nn.Module): ResNet50 with custom classification head.
    """
    
    def __init__(self, num_classes: int = 50):
        """
        Initialize the unified breed classifier.
        
        Args:
            num_classes (int): Number of breed classes. Default: 50.
        """
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
        """Forward pass through the network."""
        return self.backbone(x)


class BinaryClassifier(nn.Module):
    """
    Binary dog detection model based on ResNet18.
    
    Lightweight model for fast dog/not-dog classification.
    
    Attributes:
        backbone (nn.Module): ResNet18 with binary output.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize the binary classifier.
        
        Args:
            num_classes (int): Number of classes (2 for binary). Default: 2.
        """
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)

# =============================================================================
# MAIN UNIFIED CLASSIFIER
# =============================================================================

class UnbiasedDogClassifier:
    """
    Unbiased dog breed classifier with adaptive thresholds.
    
    Provides equal treatment for all breeds using a unified model
    architecture and per-breed confidence calibration.
    
    Attributes:
        device (torch.device): Computation device (CPU/GPU).
        binary_model (nn.Module): Dog detection model.
        breed_model (nn.Module): Unified breed classification model.
        binary_classes (list): Binary class names.
        breed_classes (list): Breed class names.
        adaptive_thresholds (dict): Per-breed confidence thresholds.
        class_metrics (dict): Per-class performance metrics.
        transform: Image preprocessing pipeline.
    """
    
    def __init__(self):
        """Initialize the classifier and load models."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f" Device: {self.device}")
        
        # Models
        self.binary_model = None
        self.breed_model = None
        
        # Classes
        self.binary_classes = ['nodog', 'dog']
        self.breed_classes = []
        
        # Adaptive thresholds per breed (initialize with defaults)
        self.adaptive_thresholds = {}
        self.default_threshold = 0.35
        
        # Per-class performance metrics
        self.class_metrics = {}
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load models and configurations
        self._load_models()
        self._load_adaptive_thresholds()
        self._load_class_metrics()
        
    def _load_models(self):
        """Load models without the selective component (bias removed)."""
        try:
            # 1. BINARY model (ResNet18)
            binary_path = "realtime_binary_models/best_model_epoch_1_acc_0.9649.pth"
            if os.path.exists(binary_path):
                logger.info(" Loading binary model (ResNet18)...")
                self.binary_model = BinaryClassifier(num_classes=2).to(self.device)
                checkpoint = torch.load(binary_path, map_location=self.device)
                self.binary_model.load_state_dict(checkpoint['model_state_dict'])
                self.binary_model.eval()
                logger.info(f" Binary model loaded - Accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
            else:
                logger.error(f" Binary model not found: {binary_path}")
                
            # 2. UNIFIED breed model (ResNet50 balanced)
            breed_path = "balanced_models/best_balanced_breed_model_epoch_20_acc_88.1366.pth"
            if os.path.exists(breed_path):
                logger.info(" Loading UNIFIED breed model (ResNet50)...")
                self.breed_model = UnifiedBreedClassifier(num_classes=50).to(self.device)
                checkpoint = torch.load(breed_path, map_location=self.device)
                self.breed_model.load_state_dict(checkpoint['model_state_dict'])
                self.breed_model.eval()
                logger.info(f" UNIFIED breed model loaded - Accuracy: {checkpoint.get('val_accuracy', 0):.2f}%")
                logger.info(f" Balanced dataset: {checkpoint.get('images_per_class', 161)} images per class")
                
                # Load breed names
                self._load_breed_names()
            else:
                logger.error(f" Breed model not found: {breed_path}")
                
        except Exception as e:
            logger.error(f" Error loading models: {e}")
            
    def _load_breed_names(self):
        """Load the names of the 50 breeds from the data directory."""
        breed_data_path = "breed_processed_data/train"
        if os.path.exists(breed_data_path):
            self.breed_classes = sorted([d for d in os.listdir(breed_data_path) 
                                       if os.path.isdir(os.path.join(breed_data_path, d))])
            logger.info(f" Loaded {len(self.breed_classes)} breeds (ALL using unified model)")
        else:
            logger.warning(" Breed directory not found, using generic names")
            self.breed_classes = [f"Breed_{i:02d}" for i in range(50)]
    
    def _load_adaptive_thresholds(self):
        """Load adaptive thresholds per breed from configuration file."""
        threshold_path = "adaptive_thresholds.json"
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path, 'r') as f:
                    self.adaptive_thresholds = json.load(f)
                logger.info(f" Loaded adaptive thresholds for {len(self.adaptive_thresholds)} breeds")
            except Exception as e:
                logger.warning(f" Error loading adaptive thresholds: {e}")
                self.adaptive_thresholds = {}
        else:
            logger.info(" Adaptive thresholds not found, using defaults")
            self.adaptive_thresholds = {}
    
    def _load_class_metrics(self):
        """
        Load per-class performance metrics.
        
        Metrics include precision, recall, and F1-score for each breed,
        used to provide confidence context in predictions.
        """
        metrics_path = "class_metrics.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    self.class_metrics = json.load(f)
                logger.info(f" Loaded metrics for {len(self.class_metrics)} breeds")
            except Exception as e:
                logger.warning(f" Error loading class metrics: {e}")
                self.class_metrics = {}
        else:
            logger.info(" Per-class metrics not found")
            self.class_metrics = {}
    
    def predict_image(self, image_path_or_pil, use_adaptive_threshold=True):
        """
        Unified classification without architectural biases.
        
        Performs two-stage classification:
        1. Binary: Detect if image contains a dog
        2. Breed: Classify breed using unified model (if dog detected)
        
        Args:
            image_path_or_pil: Path to image file or PIL Image object.
            use_adaptive_threshold (bool): Use per-breed adaptive thresholds.
            
        Returns:
            dict: Complete prediction result including:
                - is_dog: Boolean dog detection result
                - binary_confidence: Confidence of dog detection
                - breed: Predicted breed name
                - breed_confidence: Confidence score
                - breed_top5: Top 5 breed predictions with metrics
                - model_info: Architecture and bias mitigation details
        """
        try:
            # Load and process image
            if isinstance(image_path_or_pil, str):
                image = Image.open(image_path_or_pil).convert('RGB')
            else:
                image = image_path_or_pil.convert('RGB')
                
            # Transform image
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
            
            # STEP 1: BINARY classification
            if self.binary_model is not None:
                logger.info(" Iniciando clasificación binaria...")
                with torch.no_grad():
                    binary_output = self.binary_model(input_tensor)
                    binary_probs = F.softmax(binary_output, dim=1)
                    binary_confidence, binary_pred = torch.max(binary_probs, 1)
                    
                    results['binary_confidence'] = float(binary_confidence.item())
                    results['is_dog'] = bool(binary_pred.item() == 1)  # 1 = dog
                    
                    logger.info(f" Binario: {'PERRO' if results['is_dog'] else 'NO PERRO'} "
                              f"(confianza: {results['binary_confidence']:.4f})")
            else:
                logger.error(" Modelo binario es None!")
                results['error'] = "Modelo binario no disponible"
                return results
            
            # STEP 2: UNIFIED breed classification (only if a dog is detected)
            if results['is_dog']:
                logger.info(f" Iniciando clasificación de razas UNIFICADA...")
                if self.breed_model is not None and self.breed_classes:
                    with torch.no_grad():
                        breed_output = self.breed_model(input_tensor)
                        breed_probs = F.softmax(breed_output, dim=1)
                        
                        # Implementation note.
                        top5_values, top5_indices = torch.topk(breed_probs, 5, dim=1)
                        results['breed_top5'] = []
                        
                        for prob, idx in zip(top5_values[0], top5_indices[0]):
                            breed_name = self.breed_classes[idx.item()]
                            confidence = float(prob.item())
                            
                            # Get threshold adaptativo for this breed
                            if use_adaptive_threshold and breed_name in self.adaptive_thresholds:
                                threshold = self.adaptive_thresholds[breed_name]
                            else:
                                threshold = self.default_threshold
                            
                            # Implementation note.
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
                        
                        # Prediction main (Top-1)
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
                        
                        logger.info(f" Raza: {results['breed']} "
                                  f"(confianza: {results['breed_confidence']:.4f}) "
                                  f"[umbral: {results['threshold_used']:.3f}] "
                                  f"{'' if results['prediction_above_threshold'] else ''}")
                        
                        # Implementation note.
                        results['model_info'] = {
                            'architecture': 'ResNet50 Unificado',
                            'total_breeds': len(self.breed_classes),
                            'selective_bias_removed': True,
                            'all_breeds_equal_treatment': True
                        }
                        
                else:
                    logger.error(" Modelo de razas es None o no hay clases!")
                    results['error'] = "Modelo de razas no disponible"
            
            return results
            
        except Exception as e:
            logger.error(f" Error en predicción: {e}")
            return {
                'error': f"Error procesando imagen: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_class_performance(self, test_data_path=None):
        """Technical documentation in English."""
        logger.info(" EVALUANDO RENDIMIENTO POR CLASE...")
        
        if not test_data_path:
            test_data_path = "breed_processed_data/val"  # Use validation data
            
        if not os.path.exists(test_data_path):
            logger.error(f" No se encuentra el directorio de prueba: {test_data_path}")
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
            
            logger.info(f" Evaluando {breed_name}...")
            
            true_labels = []
            predicted_labels = []
            confidences = []
            
            # Evaluate all images of this breed
            image_files = [f for f in os.listdir(breed_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            correct_predictions = 0
            
            for image_file in image_files[:50]:  # Limit to 50 images per breed for speed
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
                    logger.warning(f" Error procesando {image_file}: {e}")
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
                
                logger.info(f"    {breed_name}: {breed_accuracy:.3f} accuracy "
                          f"({correct_predictions}/{len(true_labels)}) "
                          f"conf: {avg_confidence:.3f}")
        
    # Calculate overall accuracy
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        evaluation_summary = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'classes_evaluated': len(class_results),
            'class_results': class_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open('detailed_class_evaluation.json', 'w') as f:
            json.dump(evaluation_summary, f, indent=2, default=str)
        
        logger.info(f" EVALUATION COMPLETE:")
        logger.info(f"   Overall Accuracy: {overall_accuracy:.4f}")
        logger.info(f"   Samples: {total_samples}")
        logger.info(f"   Classes: {len(class_results)}")
        logger.info(f"    Saved: detailed_class_evaluation.json")
        
        return evaluation_summary
    
    def compute_adaptive_thresholds(self, validation_data_path=None):
        """
        Compute optimal confidence thresholds per breed.
        
        Uses ROC analysis to find optimal thresholds that maximize
        Youden's J statistic (TPR - FPR) for each breed.
        
        Args:
            validation_data_path (str): Path to validation data.
            
        Returns:
            dict: Per-breed optimal thresholds.
        """
        logger.info(" COMPUTING ADAPTIVE THRESHOLDS PER BREED...")
        
        if not validation_data_path:
            validation_data_path = "breed_processed_data/val"
            
        if not os.path.exists(validation_data_path):
            logger.error(f" Validation directory not found: {validation_data_path}")
            return None
        
        adaptive_thresholds = {}
        
        for breed_dir in os.listdir(validation_data_path):
            breed_path = os.path.join(validation_data_path, breed_dir)
            if not os.path.isdir(breed_path):
                continue
                
            breed_name = breed_dir
            if breed_name not in self.breed_classes:
                continue
            
            logger.info(f" Computing threshold for {breed_name}...")
            
            # Collect predictions for this breed
            true_scores = []  # Scores when image IS this breed
            false_scores = []  # Scores when image is NOT this breed
            
            # Positive images (this breed)
            image_files = [f for f in os.listdir(breed_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files[:30]:  # Limit for speed
                try:
                    image_path = os.path.join(breed_path, image_file)
                    result = self.predict_image(image_path, use_adaptive_threshold=False)
                    
                    if result.get('error') or not result['is_dog']:
                        continue
                    
                    # Find score for this breed in top5
                    for pred in result['breed_top5']:
                        if pred['breed'] == breed_name:
                            true_scores.append(pred['confidence'])
                            break
                    else:
                        # Not in top5, assign very low score
                        true_scores.append(0.01)
                        
                except Exception as e:
                    continue
            
            # Negative images (other breeds)
            other_breeds = [b for b in os.listdir(validation_data_path) 
                           if b != breed_name and os.path.isdir(os.path.join(validation_data_path, b))]
            
            for other_breed in other_breeds[:5]:  # Only a few breeds as negatives
                other_path = os.path.join(validation_data_path, other_breed)
                other_images = [f for f in os.listdir(other_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for image_file in other_images[:10]:  # Few images per breed
                    try:
                        image_path = os.path.join(other_path, image_file)
                        result = self.predict_image(image_path, use_adaptive_threshold=False)
                        
                        if result.get('error') or not result['is_dog']:
                            continue
                        
                        # Find score for target breed in top-5
                        for pred in result['breed_top5']:
                            if pred['breed'] == breed_name:
                                false_scores.append(pred['confidence'])
                                break
                        else:
                            false_scores.append(0.01)
                            
                    except Exception as e:
                        continue
            
            # Calculate optimal threshold using ROC
            if len(true_scores) > 5 and len(false_scores) > 5:
                # Prepare data for ROC
                y_true = [1] * len(true_scores) + [0] * len(false_scores)
                y_scores = true_scores + false_scores
                
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    
                    # Find threshold that maximizes Youden's J statistic (tpr - fpr)
                    j_scores = tpr - fpr
                    optimal_idx = np.argmax(j_scores)
                    optimal_threshold = thresholds[optimal_idx]
                    
                    # Constrain to reasonable range
                    optimal_threshold = max(0.1, min(0.8, optimal_threshold))
                    
                    adaptive_thresholds[breed_name] = optimal_threshold
                    
                    logger.info(f"    {breed_name}: optimal threshold = {optimal_threshold:.3f} "
                              f"(J = {j_scores[optimal_idx]:.3f})")
                    
                except Exception as e:
                    logger.warning(f"    Error computing threshold for {breed_name}: {e}")
                    adaptive_thresholds[breed_name] = self.default_threshold
            else:
                logger.warning(f"    Insufficient data for {breed_name}, using default threshold")
                adaptive_thresholds[breed_name] = self.default_threshold
        
        # Save adaptive thresholds
        with open('adaptive_thresholds.json', 'w') as f:
            json.dump(adaptive_thresholds, f, indent=2)
        
        self.adaptive_thresholds = adaptive_thresholds
        
        logger.info(f" ADAPTIVE THRESHOLDS COMPUTED:")
        logger.info(f"   Breeds processed: {len(adaptive_thresholds)}")
        logger.info(f"   Threshold range: {min(adaptive_thresholds.values()):.3f} - {max(adaptive_thresholds.values()):.3f}")
        logger.info(f"    Saved: adaptive_thresholds.json")
        
        return adaptive_thresholds
    
    def get_model_info(self):
        """
        Get comprehensive model architecture and configuration info.
        
        Returns:
            dict: Model architecture, bias mitigation status, and metrics.
        """
        return {
            'architecture': 'Unified - ResNet50 for all breeds',
            'binary_model_loaded': self.binary_model is not None,
            'breed_model_loaded': self.breed_model is not None,
            'selective_model_removed': True,  # Bias removed
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
# FLASK API ENDPOINTS
# =============================================================================

app = Flask(__name__)
classifier = UnbiasedDogClassifier()


@app.route('/')
def index():
    """
    Serve the main web interface for the unbiased classifier.
    
    Returns:
        str: HTML template with upload form and bias mitigation info.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title> Clasificador Unificado Sin Sesgos</title>
        <style>
            body { font-family: Arial; margin: 40px; background: # f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; color: # 2c3e50; }
            .bias-info { background: # e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .improvement { color: # 27ae60; font-weight: bold; }
            .upload-area { border: 2px dashed # 3498db; padding: 40px; text-align: center; margin: 20px 0; }
            input[type="file"] { margin: 10px; }
            button { background: # 3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .results { background: # f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> Clasificador Unificado de Perros</h1>
                <h2> Versión Sin Sesgos Arquitecturales</h2>
            </div>
            
            <div class="bias-info">
                <h3> Mejoras Implementadas:</h3>
                <ul>
                    <li class="improvement"> Modelo selectivo eliminado (todas las razas tienen igual tratamiento)</li>
                    <li class="improvement"> Arquitectura unificada ResNet50</li>
                    <li class="improvement"> Umbrales adaptativos por raza</li>
                    <li class="improvement"> Métricas detalladas por clase</li>
                </ul>
            </div>
            
            <div class="upload-area">
                <h3> Subir Imagen de Perro</h3>
                <input type="file" id="fileInput" accept="image/*">
                <br><button onclick="analyzeImage()"> Analizar</button>
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
                        <h3> Resultados del Análisis Sin Sesgos</h3>
                        <p><strong> Es perro:</strong> ${data.is_dog ? ' Sí' : ' No'} (${(data.binary_confidence * 100).toFixed(1)}%)</p>
                        ${data.breed ? `
                            <p><strong> Raza:</strong> ${data.breed}</p>
                            <p><strong> Confianza:</strong> ${(data.breed_confidence * 100).toFixed(1)}%</p>
                            <p><strong> Umbral usado:</strong> ${data.threshold_used ? data.threshold_used.toFixed(3) : 'N/A'}</p>
                            <p><strong> Sobre umbral:</strong> ${data.prediction_above_threshold ? 'Sí' : 'No'}</p>
                            <p><strong> Arquitectura:</strong> ${data.model_architecture}</p>
                            <p><strong> Mejora:</strong> ${data.bias_mitigation}</p>
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
    """
    API endpoint for unbiased dog breed prediction.
    
    Accepts an image file upload and returns comprehensive
    classification results using the unified model.
    
    Returns:
        JSON: Prediction result with breed and confidence scores.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image found in request'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(file.read()))
        
        # Perform prediction with unified model
        result = classifier.predict_image(image, use_adaptive_threshold=True)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})


@app.route('/info')
def model_info():
    """
    API endpoint for model architecture information.
    
    Returns:
        JSON: Model configuration and bias mitigation status.
    """
    return jsonify(classifier.get_model_info())


@app.route('/evaluate')
def evaluate():
    """
    API endpoint for detailed per-class performance evaluation.
    
    Returns:
        JSON: Accuracy and confidence metrics per breed.
    """
    results = classifier.evaluate_class_performance()
    return jsonify(results)


@app.route('/compute_thresholds')
def compute_thresholds():
    """
    API endpoint to compute adaptive thresholds per breed.
    
    Returns:
        JSON: Computed optimal thresholds for each breed.
    """
    results = classifier.compute_adaptive_thresholds()
    return jsonify({'adaptive_thresholds': results})

if __name__ == "__main__":
    print("" * 80)
    print(" UNBIASED UNIFIED DOG CLASSIFIER")
    print("" * 80)
    print(" Implemented improvements:")
    print("    Selective model REMOVED")
    print("    UNIFIED architecture (ResNet50)")
    print("    ADAPTIVE thresholds per breed")
    print("    DETAILED metrics per class")
    print("" * 80)
    
    info = classifier.get_model_info()
    print(" Model status:")
    print(f"   Binary loaded: {'' if info['binary_model_loaded'] else ''}")
    print(f"   Breeds loaded: {'' if info['breed_model_loaded'] else ''}")
    print(f"   Bias removed: {'' if info['selective_model_removed'] else ''}")
    print(f"   Available breeds: {info['num_breeds']}")
    
    print(f"\n Starting server at: http://localhost:5001")
    app.run(host='127.0.0.1', port=5001, debug=False)