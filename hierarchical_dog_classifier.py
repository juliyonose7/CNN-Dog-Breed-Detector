#!/usr/bin/env python3
"""
Hierarchical Dog Classifier with Web Interface
===============================================

This module implements a complete hierarchical dog classification system combining
two trained models with a Flask-based web interface.

Architecture:
    1. Binary Model (ResNet18): Detects if an image contains a dog or not
    2. Breed Model (ResNet50): Identifies among 50 dog breeds
    3. Selective Model (ResNet34): Specializes in 6 problematic breeds

Features:
    - Temperature scaling for prediction calibration
    - Selective model fallback for problematic breeds
    - Interactive web frontend with drag-and-drop
    - Real-time temperature adjustment via API
    - Comprehensive logging for debugging

Models Handled:
    - Binary: ResNet18 for dog/not-dog classification
    - Balanced Breed: ResNet50 trained on balanced dataset (161 images/class)
    - Selective: ResNet34 for basset, beagle, Labrador, Norwegian elkhound, pug, Samoyed

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL ARCHITECTURE DEFINITIONS
# =============================================================================

class FastBinaryModel(nn.Module):
    """
    Fast binary classification model based on ResNet18.
    
    Lightweight architecture optimized for quick dog/not-dog inference.
    
    Args:
        num_classes (int): Number of output classes (default: 2).
    
    Architecture:
        - Backbone: ResNet18 (weights loaded separately)
        - Output: Linear layer with num_classes outputs
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the binary classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, 224, 224).
        
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes).
        """
        return self.backbone(x)


class BreedModel(nn.Module):
    """
    Breed classification model based on ResNet34.
    
    Standard breed classifier for 50-class dog breed identification.
    
    Args:
        num_classes (int): Number of breed classes (default: 50).
    
    Architecture:
        - Backbone: ResNet34 (weights loaded separately)
        - Output: Linear layer with num_classes outputs
    """
    
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the breed classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, 224, 224).
        
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes).
        """
        return self.backbone(x)

# =============================================================================
# HIERARCHICAL CLASSIFIER
# =============================================================================

class HierarchicalDogClassifier:
    """
    Hierarchical dog classifier with multi-model ensemble.
    
    Combines binary, breed, and selective models for comprehensive dog
    classification with temperature scaling for calibration.
    
    Attributes:
        device (torch.device): Computation device (cuda/cpu).
        binary_model: ResNet18 for dog/not-dog detection.
        breed_model: ResNet50 for 50-breed classification.
        selective_model: ResNet34 for 6 problematic breeds.
        binary_classes (list): ['nodog', 'dog'].
        breed_classes (list): List of 50 breed names.
        breed_temperature (float): Temperature for breed prediction softening.
        binary_temperature (float): Temperature for binary prediction.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f" Device: {self.device}")
        
        # Model references
        self.binary_model = None
        self.breed_model = None
        self.selective_model = None
        self.selective_classes = {}
        self.selective_idx_to_breed = {}
        
        # Class labels
        self.binary_classes = ['nodog', 'dog']
        self.breed_classes = []
        
        # Temperature scaling parameters for prediction calibration
        self.breed_temperature = 10.0  # Higher = softer probabilities
        self.binary_temperature = 1.0  # Keep binary predictions sharp
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load all models
        self._load_models()
        
    def _load_models(self):
        """
        Load all trained models for hierarchical classification.
        
        Loads binary, breed, and selective models from their respective
        checkpoint files. Handles different model architectures appropriately.
        """
        try:
            # 1. BINARY MODEL (ResNet18)
            binary_path = "realtime_binary_models/best_model_epoch_1_acc_0.9649.pth"
            if os.path.exists(binary_path):
                logger.info(" Loading binary model (ResNet18)...")
                self.binary_model = FastBinaryModel(num_classes=2).to(self.device)
                checkpoint = torch.load(binary_path, map_location=self.device)
                self.binary_model.load_state_dict(checkpoint['model_state_dict'])
                self.binary_model.eval()
                logger.info(f" Binary model loaded - Accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
            else:
                logger.error(f" Binary model not found: {binary_path}")
                
            # 2. BALANCED BREED MODEL (ResNet50) - Primary breed classifier
            breed_path = "balanced_models/best_balanced_breed_model_epoch_20_acc_88.1366.pth"
            if os.path.exists(breed_path):
                logger.info(" Loading BALANCED breed model (ResNet50)...")
                
                # Define balanced model architecture (ResNet50 with dropout)
                class BalancedBreedClassifier(nn.Module):
                    """
                    Balanced breed classifier with dropout regularization.
                    
                    Custom architecture with additional dropout layers and
                    intermediate fully connected layer for better generalization.
                    """
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
                logger.info(f" BALANCED breed model loaded - Accuracy: {checkpoint.get('val_accuracy', 0):.2f}%")
                logger.info(f" Balanced dataset: {checkpoint.get('images_per_class', 0)} images per class")
                
                # Load breed names from training directory
                self._load_breed_names()
            else:
                logger.warning(f" Balanced model not found, trying original model...")
                # Fallback to original model if balanced not available
                breed_path_original = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
                if os.path.exists(breed_path_original):
                    logger.info(" Loading original breed model (ResNet34)...")
                    self.breed_model = BreedModel(num_classes=50).to(self.device)
                    checkpoint = torch.load(breed_path_original, map_location=self.device)
                    self.breed_model.load_state_dict(checkpoint['model_state_dict'])
                    self.breed_model.eval()
                    logger.info(f" Original breed model loaded - Accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
                    self._load_breed_names()
                else:
                    logger.error(f" Neither balanced nor original model found")
                
            # 3. SELECTIVE MODEL (ResNet34) - For problematic breeds
            self.selective_model = None
            self.selective_classes = {}
            selective_path = "selective_models/best_selective_model.pth"
            
            if os.path.exists(selective_path):
                logger.info(" Loading selective model (6 problematic breeds)...")
                
                # Define selective model architecture
                class SelectiveBreedClassifier(nn.Module):
                    """
                    Specialized classifier for problematic breeds.
                    
                    Trained specifically on breeds that the main model
                    struggles with to provide fallback predictions.
                    """
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
                
                # Build class mapping for selective model
                self.selective_classes = checkpoint['class_to_idx']
                self.selective_idx_to_breed = {v: k for k, v in self.selective_classes.items()}
                
                logger.info(f" Selective model loaded - Accuracy: {checkpoint.get('val_accuracy', 0):.2f}%")
                logger.info(f" Problematic breeds: {list(self.selective_classes.keys())}")
            else:
                logger.warning(" Selective model not found, using main model only")
                
        except Exception as e:
            logger.error(f" Error loading models: {e}")
            
    def _load_breed_names(self):
        """
        Load breed names from the training data directory.
        
        Reads subdirectory names from the breed training folder to get
        the list of 50 breed class names.
        """
        breed_data_path = "breed_processed_data/train"
        if os.path.exists(breed_data_path):
            self.breed_classes = sorted([d for d in os.listdir(breed_data_path) 
                                       if os.path.isdir(os.path.join(breed_data_path, d))])
            logger.info(f" Loaded {len(self.breed_classes)} breed names")
        else:
            logger.warning(" Breed directory not found, using generic names")
            self.breed_classes = [f"Breed_{i:02d}" for i in range(50)]
    
    def predict_image(self, image_path_or_pil, confidence_threshold=0.5):
        """
        Perform complete hierarchical prediction on an image.
        
        Executes two-stage classification:
        1. Binary classification to detect if image contains a dog
        2. Breed classification if dog detected (with optional selective model fallback)
        
        Args:
            image_path_or_pil: Path to image file or PIL Image object.
            confidence_threshold (float): Minimum confidence for breed classification.
        
        Returns:
            dict: Complete prediction results containing:
                - timestamp: ISO format timestamp
                - image_size: (width, height) tuple
                - is_dog: Boolean indicating dog detection
                - binary_confidence: Confidence of binary prediction
                - breed: Predicted breed name (or None)
                - breed_confidence: Confidence of breed prediction
                - breed_top3: List of top-3 breed predictions
                - temperature: Temperature value used
                - error: Error message if any
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
                'breed_top3': [],
                'temperature': self.breed_temperature,
                'error': None
            }
            
            # STEP 1: BINARY classification
            if self.binary_model is not None:
                logger.info(" Starting binary classification...")
                with torch.no_grad():
                    binary_output = self.binary_model(input_tensor)
                    # Apply temperature scaling to binary model
                    binary_probs = F.softmax(binary_output / self.binary_temperature, dim=1)
                    binary_confidence, binary_pred = torch.max(binary_probs, 1)
                    
                    results['binary_confidence'] = float(binary_confidence.item())
                    results['is_dog'] = bool(binary_pred.item() == 1)  # 1 = dog
                    
                    logger.info(f" Binary: {'DOG' if results['is_dog'] else 'NOT DOG'} "
                              f"(confidence: {results['binary_confidence']:.4f})")
            else:
                logger.error(" Binary model is None!")
                results['error'] = "Binary model not available"
                return results
            
            # STEP 2: BREED classification (only if dog detected)
            if results['is_dog'] and results['binary_confidence'] >= confidence_threshold:
                logger.info(f" Starting breed classification (confidence: {results['binary_confidence']:.4f} >= {confidence_threshold})")
                if self.breed_model is not None and self.breed_classes:
                    logger.info(f" Breed model available, {len(self.breed_classes)} breeds loaded")
                    with torch.no_grad():
                        breed_output = self.breed_model(input_tensor)
                        # Apply temperature scaling to soften breed predictions
                        breed_probs = F.softmax(breed_output / self.breed_temperature, dim=1)
                        
                        # Top-1 main prediction
                        breed_confidence, breed_pred = torch.max(breed_probs, 1)
                        main_breed = self.breed_classes[breed_pred.item()]
                        main_confidence = float(breed_confidence.item())
                        
                        # Check if selective model should be used for problematic breeds
                        # These breeds have historically shown confusion
                        problematic_breeds = ['basset', 'beagle', 'Labrador_retriever', 'Norwegian_elkhound', 'pug', 'Samoyed']
                        use_selective = False
                        
                        if self.selective_model is not None and main_breed in problematic_breeds:
                            logger.info(f" Problematic breed detected: {main_breed}, using selective model...")
                            use_selective = True
                        elif self.selective_model is not None and main_confidence < 0.15:
                            # Low confidence - try selective model
                            logger.info(f" Low confidence ({main_confidence:.4f}), trying selective model...")
                            use_selective = True
                        
                        if use_selective:
                            # Use selective model for problematic breeds
                            selective_output = self.selective_model(input_tensor)
                            selective_probs = F.softmax(selective_output / self.breed_temperature, dim=1)
                            selective_confidence, selective_pred = torch.max(selective_probs, 1)
                            selective_breed = self.selective_idx_to_breed[selective_pred.item()]
                            selective_conf = float(selective_confidence.item())
                            
                            # Use selective prediction if 20% more confident than main model
                            if selective_conf > main_confidence * 1.2:  # 20% advantage for selective model
                                logger.info(f" Using selective model: {selective_breed} (conf: {selective_conf:.4f})")
                                results['breed'] = selective_breed
                                results['breed_confidence'] = selective_conf
                                results['model_used'] = 'selective'
                                
                                # Top-3 from selective model
                                top3_values, top3_indices = torch.topk(selective_probs, min(3, len(self.selective_classes)), dim=1)
                                results['breed_top3'] = [
                                    {
                                        'breed': self.selective_idx_to_breed[idx.item()],
                                        'confidence': float(prob.item())
                                    }
                                    for prob, idx in zip(top3_values[0], top3_indices[0])
                                ]
                            else:
                                logger.info(f" Main model better: {main_breed} (conf: {main_confidence:.4f})")
                                results['breed'] = main_breed
                                results['breed_confidence'] = main_confidence
                                results['model_used'] = 'main'
                                
                                # Top-3 from main model
                                top3_values, top3_indices = torch.topk(breed_probs, 3, dim=1)
                                results['breed_top3'] = [
                                    {
                                        'breed': self.breed_classes[idx.item()],
                                        'confidence': float(prob.item())
                                    }
                                    for prob, idx in zip(top3_values[0], top3_indices[0])
                                ]
                        else:
                            # Use only main model
                            results['breed'] = main_breed
                            results['breed_confidence'] = main_confidence
                            results['model_used'] = 'main'
                            
                            # Top-3 predictions from main model
                            top3_values, top3_indices = torch.topk(breed_probs, 3, dim=1)
                            results['breed_top3'] = [
                                {
                                    'breed': self.breed_classes[idx.item()],
                                    'confidence': float(prob.item())
                                }
                                for prob, idx in zip(top3_values[0], top3_indices[0])
                            ]
                        
                        logger.info(f" Breed: {results['breed']} "
                                  f"(confidence: {results['breed_confidence']:.4f}) "
                                  f"[{results.get('model_used', 'main')}]")
                else:
                    logger.error(" Breed model is None or no classes!")
                    logger.error(f" breed_model: {self.breed_model is not None}, breed_classes: {len(self.breed_classes) if self.breed_classes else 0}")
                    results['error'] = "Breed model not available"
            elif results['is_dog'] and results['binary_confidence'] < confidence_threshold:
                logger.info(f" Dog detected but with low confidence ({results['binary_confidence']:.4f} < {confidence_threshold})")
                # Dog detected but with low confidence - attempt breed prediction anyway
                if self.breed_model is not None and self.breed_classes:
                    logger.info(" Attempting breed classification with low confidence...")
                    with torch.no_grad():
                        breed_output = self.breed_model(input_tensor)
                        # Apply temperature scaling
                        breed_probs = F.softmax(breed_output / self.breed_temperature, dim=1)
                        
                        # Top-1 prediction
                        breed_confidence, breed_pred = torch.max(breed_probs, 1)
                        results['breed'] = f"Possibly: {self.breed_classes[breed_pred.item()]}"
                        results['breed_confidence'] = float(breed_confidence.item())
                        
                        # Top-3 predictions
                        top3_values, top3_indices = torch.topk(breed_probs, 3, dim=1)
                        results['breed_top3'] = [
                            {
                                'breed': self.breed_classes[idx.item()],
                                'confidence': float(prob.item())
                            }
                            for prob, idx in zip(top3_values[0], top3_indices[0])
                        ]
                        
                        logger.info(f" Breed (low confidence): {results['breed']} "
                                  f"(confidence: {results['breed_confidence']:.4f})")
                else:
                    results['breed'] = "Insufficient confidence to determine breed"
                
            return results
            
        except Exception as e:
            logger.error(f" Error in prediction: {e}")
            return {
                'error': f"Error processing image: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self):
        """
        Get comprehensive information about loaded models.
        
        Returns:
            dict: Model status, architectures, and configuration parameters.
        """
        return {
            'binary_model_loaded': self.binary_model is not None,
            'breed_model_loaded': self.breed_model is not None,
            'selective_model_loaded': self.selective_model is not None,
            'binary_architecture': 'ResNet18',
            'breed_architecture': 'ResNet50 (Balanced)',
            'selective_architecture': 'ResNet34 (6 breeds)',
            'num_breeds': len(self.breed_classes),
            'num_selective_breeds': len(self.selective_classes),
            'device': str(self.device),
            'breed_classes': self.breed_classes[:10],  # Show only first 10 to avoid overload
            'selective_breeds': list(self.selective_classes.keys()) if self.selective_classes else [],
            'breed_temperature': self.breed_temperature,
            'binary_temperature': self.binary_temperature,
            'dataset_balanced': True,
            'images_per_class': 161
        }
    
    def adjust_temperature(self, breed_temp=None, binary_temp=None):
        """
        Adjust temperature scaling parameters for prediction calibration.
        
        Higher temperatures produce softer probability distributions,
        while lower temperatures produce more confident predictions.
        
        Args:
            breed_temp (float, optional): New temperature for breed predictions.
            binary_temp (float, optional): New temperature for binary predictions.
        """
        if breed_temp is not None:
            self.breed_temperature = breed_temp
            logger.info(f" Breed temperature adjusted to: {breed_temp}")
        if binary_temp is not None:
            self.binary_temperature = binary_temp 
            logger.info(f" Binary temperature adjusted to: {binary_temp}")

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)
classifier = HierarchicalDogClassifier()

# HTML Template for the frontend (Spanish interface)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title> Hierarchical Dog Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, # 667eea 0%, #764ba2 100%);
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
            background: linear-gradient(135deg, # ff6b6b, #ffa500);
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
            border: 3px dashed # ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: # ff6b6b;
            background: # fafafa;
        }
        
        .upload-area.dragover {
            border-color: # ff6b6b;
            background: # fff5f5;
        }
        
        # fileInput {
            display: none;
        }
        
        .upload-icon {
            font-size: 4em;
            color: # ddd;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.3em;
            color: # 666;
            margin-bottom: 15px;
        }
        
        .upload-subtext {
            color: # 999;
            font-size: 0.9em;
        }
        
        .btn {
            background: linear-gradient(135deg, # ff6b6b, #ffa500);
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
            background: # f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
        }
        
        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid # eee;
        }
        
        .result-item:last-child {
            border-bottom: none;
        }
        
        .result-label {
            font-weight: bold;
            color: # 333;
        }
        
        .result-value {
            color: # 666;
        }
        
        .confidence-bar {
            width: 200px;
            height: 10px;
            background: # eee;
            border-radius: 5px;
            overflow: hidden;
            margin-left: 15px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, # ff6b6b, #ffa500);
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
            border-bottom: 1px solid # eee;
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
            border: 4px solid # f3f3f3;
            border-top: 4px solid # ff6b6b;
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
            background: # ffebee;
            color: # c62828;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid # c62828;
        }
        
        .success {
            background: # e8f5e8;
            color: # 2e7d32;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid # 2e7d32;
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
            <h1> Dog Classifier</h1>
            <p>Hierarchical AI System • ResNet18 + ResNet34</p>
        </div>
        
        <div class="content">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon"></div>
                <div class="upload-text">Click here or drag an image</div>
                <div class="upload-subtext">Formats: JPG, PNG, GIF • Max 10MB</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="preview-container">
                <img class="preview-image" id="previewImage" alt="Preview">
            </div>
            
            <div style="text-align: center;">
                <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
                     Analyze Image
                </button>
                
                <div style="margin-top: 20px; padding: 15px; background: # f8f9fa; border-radius: 10px;">
                    <label for="tempSlider" style="display: block; margin-bottom: 10px; font-weight: bold;">
                         Calibration Temperature: <span id="tempValue">10.0</span>
                    </label>
                    <input type="range" id="tempSlider" min="1" max="15" step="0.5" value="10.0" 
                           style="width: 100%; margin-bottom: 10px;" onchange="updateTemperature()">
                    <div style="font-size: 0.9em; color: # 666;">
                        Lower = More extreme | Higher = More balanced
                    </div>
                </div>
            </div>
            
            <div class="loading">
                <div class="spinner"></div>
                <div>Analyzing image with AI...</div>
            </div>
            
            <div class="results" id="results">
                <!-- Results will be displayed here -->
            </div>
        </div>
    </div>

    <script>
        let selectedImage = null;
        
        // Set up drag & drop events
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
                showError('Please select a valid image file.');
                return;
            }
            
            if (file.size > 10 * 1024 * 1024) {
                showError('File is too large. Maximum 10MB.');
                return;
            }
            
            selectedImage = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.style.display = 'block';
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);
            
            // Hide previous results
            resultsDiv.style.display = 'none';
        }
        
        async function analyzeImage() {
            if (!selectedImage) return;
            
            // Show loading
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
                showError('Error connecting to server: ' + error.message);
            } finally {
                loadingDiv.style.display = 'none';
                analyzeBtn.disabled = false;
            }
        }
        
        function showResults(data) {
            let html = '<h3> Analysis Results</h3>';
            
            // Binary result
            html += `
                <div class="result-item">
                    <div class="result-label"> Detection:</div>
                    <div style="display: flex; align-items: center;">
                        <span class="result-value">${data.is_dog ? ' IS A DOG' : ' NOT A DOG'}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${data.binary_confidence * 100}%"></div>
                        </div>
                        <span style="margin-left: 10px; color: # 666;">${(data.binary_confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
            
            // Breed result if dog is detected
            if (data.is_dog && data.breed) {
                if (data.breed_confidence > 0) {
                    html += `
                        <div class="result-item">
                            <div class="result-label"> Main Breed:</div>
                            <div style="display: flex; align-items: center;">
                                <span class="result-value">${data.breed}</span>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${data.breed_confidence * 100}%"></div>
                                </div>
                                <span style="margin-left: 10px; color: # 666;">${(data.breed_confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="result-item">
                            <div class="result-label"> Breed:</div>
                            <span class="result-value">${data.breed}</span>
                        </div>
                    `;
                }
                
                // Top 3 breeds
                if (data.breed_top3 && data.breed_top3.length > 0) {
                    html += `
                        <div class="result-item">
                            <div class="result-label"> Top 3 Breeds:</div>
                            <div class="breed-list">
                    `;
                    
                    data.breed_top3.forEach((breed, index) => {
                        const medal = ['', '', ''][index];
                        html += `
                            <div class="breed-item">
                                <span>${medal} ${breed.breed}</span>
                                <div style="display: flex; align-items: center;">
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${breed.confidence * 100}%"></div>
                                    </div>
                                    <span style="margin-left: 10px; color: # 666;">${(breed.confidence * 100).toFixed(1)}%</span>
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
            
            // Technical information
            html += `
                <div class="result-item">
                    <div class="result-label"> Models:</div>
                    <span class="result-value">ResNet18 (Binary) + ResNet34 (Breeds)</span>
                </div>
                <div class="result-item">
                    <div class="result-label"> Temperature:</div>
                    <span class="result-value">${data.temperature || 'N/A'}</span>
                </div>
                <div class="result-item">
                    <div class="result-label"> Processed:</div>
                    <span class="result-value">${new Date(data.timestamp).toLocaleTimeString()}</span>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
        
        function showError(message) {
            resultsDiv.innerHTML = `<div class="error"> ${message}</div>`;
            resultsDiv.style.display = 'block';
        }
        
        function updateTemperature() {
            const slider = document.getElementById('tempSlider');
            const tempValue = document.getElementById('tempValue');
            tempValue.textContent = slider.value;
            
            // Send new temperature to server
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
                      console.log(' Temperatura ajustada:', data.breed_temperature);
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
    """Main frontend page serving the interactive web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/test')
def test():
    """
    Test endpoint to verify server status.
    
    Returns:
        JSON: Status message with timestamp.
    """
    logger.info(" Test endpoint called")
    return jsonify({
        'status': 'ok',
        'message': 'Server running correctly',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for image prediction.
    
    Accepts image file upload and returns hierarchical classification results.
    
    Request:
        POST /predict
        Content-Type: multipart/form-data
        Body: image (file)
    
    Returns:
        JSON: Complete prediction results including dog detection and breed.
    """
    logger.info(" Prediction request received")
    try:
        logger.info(" Checking files in request...")
        if 'image' not in request.files:
            logger.error(" No image in request")
            return jsonify({'error': 'No image found in request'})
        
        file = request.files['image']
        logger.info(f" File received: {file.filename}")
        if file.filename == '':
            logger.error(" Empty filename")
            return jsonify({'error': 'No file selected'})
        
        logger.info(" Processing image...")
        # Convert to PIL Image
        image = Image.open(io.BytesIO(file.read()))
        logger.info(f" Image loaded: {image.size}")
        
        # Run hierarchical prediction
        logger.info(" Starting prediction...")
        result = classifier.predict_image(image, confidence_threshold=0.35)
        logger.info(f" Result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f" API error: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/info')
def model_info():
    """
    Get detailed model information.
    
    Returns:
        JSON: Complete model status and configuration.
    """
    return jsonify(classifier.get_model_info())

@app.route('/health')
def health_check():
    """
    Health check endpoint for service monitoring.
    
    Returns:
        JSON: Service status with model availability.
    """
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
    """
    Dynamically adjust temperature scaling parameters.
    
    Request:
        POST /adjust_temp
        Content-Type: application/json
        Body: {"breed_temperature": float, "binary_temperature": float}
    
    Returns:
        JSON: Updated temperature values.
    """
    try:
        data = request.get_json()
        breed_temp = data.get('breed_temperature')
        binary_temp = data.get('binary_temperature')
        
        classifier.adjust_temperature(breed_temp, binary_temp)
        
        return jsonify({
            'status': 'success',
            'message': 'Temperatures adjusted',
            'breed_temperature': classifier.breed_temperature,
            'binary_temperature': classifier.binary_temperature
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the hierarchical dog classifier system.
    
    Initializes models, displays status information, and starts the Flask web server.
    """
    print("" * 80)
    print(" HIERARCHICAL DOG CLASSIFIER - INTEGRATED SYSTEM")
    print("" * 80)
    print(" Models:")
    print("    Binary: ResNet18 (dog/not dog)")
    print("    Breeds: ResNet50 BALANCED (50 breeds)")
    print("    Selective: ResNet34 (6 problematic breeds)")
    print("" * 80)
    
    # Display model status
    info = classifier.get_model_info()
    print(" Model Status:")
    print(f"   Binary loaded: {'' if info['binary_model_loaded'] else ''}")
    print(f"   Breeds loaded: {'' if info['breed_model_loaded'] else ''}")
    print(f"   Selective loaded: {'' if info['selective_model_loaded'] else ''}")
    print(f"   Device: {info['device']}")
    print(f"   Available breeds: {info['num_breeds']}")
    print(f"   Balanced dataset:  ({info['images_per_class']} img/breed)")
    if info['selective_model_loaded']:
        print(f"   Selective breeds: {info['num_selective_breeds']} ({', '.join(info['selective_breeds'])})")
    
    if not info['binary_model_loaded'] or not info['breed_model_loaded']:
        print("\n WARNING: Some models are not loaded")
        print("   Verify that these files exist:")
        print("   - binary_models/best_fast_binary_model.pth")
        print("   - autonomous_breed_models/best_breed_model.pth")
    
    print("\n Starting web server...")
    print(" Open your browser at: http://localhost:5000")
    print("" * 80)
    
    # Start Flask server with CORS enabled
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == "__main__":
    main()