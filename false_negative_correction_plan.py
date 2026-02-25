#!/usr/bin/env python3
"""
False Negative Correction Plan Module
=====================================

This module provides comprehensive strategies for correcting false negatives
in a 119-class dog breed classification model. It identifies problematic breeds
and implements various correction techniques including threshold adjustment,
weighted loss functions, data augmentation, and ensemble methods.

Features:
    - Identification of breeds with high false negative rates
    - Adaptive threshold adjustment per class
    - Weighted and focal loss function implementations
    - Breed-specific data augmentation strategies
    - Ensemble methods for improved recall
    - Implementation roadmap with phased approach
    - Quick-fix scripts for immediate deployment

Author: AI System
Date: 2024
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class FalseNegativeCorrector:
    """
    A corrector class for reducing false negatives in breed classification.
    
    This class implements multiple strategies to improve recall for breeds
    that have high false negative rates. Breeds are categorized by priority
    based on their false negative impact.
    
    Attributes:
        problematic_breeds (dict): Dictionary categorizing breeds by priority level:
            - 'critical': Breeds with highest false negative rates (>40%)
            - 'high_priority': Breeds with significant false negative rates (30-40%)
            - 'medium_priority': Breeds with moderate false negative rates (20-30%)
    """
    
    def __init__(self):
        self.problematic_breeds = {
            'critical': ['Lhasa', 'cairn'],
            'high_priority': ['Siberian_husky', 'whippet', 'malamute', 'Australian_terrier', 
                            'Norfolk_terrier', 'toy_terrier', 'Italian_greyhound'],
            'medium_priority': ['Lakeland_terrier', 'Border_terrier', 'bluetick', 
                              'Rhodesian_ridgeback', 'Ibizan_hound']
        }
        
    def generate_correction_plan(self):
        """
        Generate a comprehensive correction plan for false negatives.
        
        Creates a dictionary of correction strategies including threshold adjustment,
        weighted loss, data augmentation, focal loss, and ensemble methods.
        
        Returns:
            dict: Dictionary containing all correction strategies with their
                  implementations, expected improvements, and difficulty levels.
        """
        print("🛠️ FALSE NEGATIVE CORRECTION PLAN")
        print("=" * 60)
        
        correction_strategies = {
            "1_threshold_adjustment": self.threshold_adjustment_strategy(),
            "2_weighted_loss": self.weighted_loss_strategy(), 
            "3_data_augmentation": self.data_augmentation_strategy(),
            "4_focal_loss": self.focal_loss_strategy(),
            "5_ensemble_methods": self.ensemble_strategy(),
            "6_hard_negative_mining": self.hard_negative_mining_strategy(),
            "7_class_balancing": self.class_balancing_strategy(),
            "8_feature_enhancement": self.feature_enhancement_strategy()
        }
        
        return correction_strategies
    
    def threshold_adjustment_strategy(self):
        """
        Strategy 1: Per-class threshold adjustment.
        
        Implements adaptive thresholds that are lower for breeds with high
        false negative rates, making the model less conservative for these classes.
        
        Returns:
            dict: Strategy details including implementation code, expected
                  improvement (15-25%), and low difficulty level.
        """
        print("\n📈 STRATEGY 1: PER-CLASS THRESHOLD ADJUSTMENT")
        print("-" * 50)
        print("🎯 Objective: Reduce thresholds for conservative breeds")
        
        strategy = {
            "description": "Use adaptive lower thresholds for breeds with many false negatives",
            "implementation": """
# Adaptive thresholds for problematic breeds
BREED_THRESHOLDS = {
    'Lhasa': 0.35,           # Very low (was conservative)
    'cairn': 0.40,           # Low (was very conservative)
    'Siberian_husky': 0.45,  # Low-medium
    'whippet': 0.45,         # Low-medium
    'malamute': 0.50,        # Medium
    'Australian_terrier': 0.50,
    'Norfolk_terrier': 0.50,
    'toy_terrier': 0.55,     # Close to standard
    # Standard breeds remain at default threshold
}

def apply_adaptive_thresholds(predictions, breed_names, default_threshold=0.60):
    adjusted_predictions = []
    
    for i, breed in enumerate(breed_names):
        threshold = BREED_THRESHOLDS.get(breed, default_threshold)
        pred_score = predictions[i]
        
        # Apply per-breed customized threshold
        if pred_score >= threshold:
            adjusted_predictions.append((breed, pred_score, True))
        else:
            adjusted_predictions.append((breed, pred_score, False))
    
    return adjusted_predictions
            """,
            "expected_improvement": "15-25% reduction in false negatives",
            "difficulty": "LOW - no retraining required"
        }
        
        print("📊 Expected improvement: 15-25% fewer false negatives")
        
        return strategy
    
    def weighted_loss_strategy(self):
        """
        Strategy 2: Weighted loss function implementation.
        
        Implements class-weighted loss that penalizes false negatives more
        heavily for problematic breeds, encouraging the model to improve recall.
        
        Returns:
            dict: Strategy details including WeightedFocalLoss implementation,
                  expected improvement (20-35%), and medium difficulty level.
        """
        print("\n🎯 STRATEGY 2: WEIGHTED LOSS FUNCTION")
        print("-" * 50)
        print("🎯 Objective: Penalize false negatives more than false positives")
        
        strategy = {
            "description": "Use class weights that penalize false negatives more for problematic breeds",
            "implementation": """
import torch.nn as nn

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, recall_weight=2.0):
        super().__init__()
        self.alpha = alpha  # Per-class weights
        self.gamma = gamma  # Focal loss factor
        self.recall_weight = recall_weight  # Extra penalty for false negatives
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha)(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Focal loss component
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Extra penalty for false negatives (misclassified samples)
        pred_classes = torch.argmax(inputs, dim=1)
        false_negatives = (pred_classes != targets)
        
        # Apply additional penalty to false negative cases
        penalty = torch.where(false_negatives, 
                            torch.tensor(self.recall_weight), 
                            torch.tensor(1.0)).to(inputs.device)
        
        return (focal_loss * penalty).mean()

# Class weights for problematic breeds
CLASS_WEIGHTS = {
    'Lhasa': 3.0,           # Very high weight (was 46% FN)
    'cairn': 2.8,           # Very high weight (was 41% FN)
    'Siberian_husky': 2.5,  # High weight
    'whippet': 2.3,         # High weight
    'malamute': 2.2,        # Medium-high weight
    # Normal breeds = 1.0
}

def create_class_weights(num_classes, problematic_breeds_weights):
    weights = torch.ones(num_classes)
    
    for breed_idx, breed_name in enumerate(breed_names):
        if breed_name in problematic_breeds_weights:
            weights[breed_idx] = problematic_breeds_weights[breed_name]
    
    return weights
            """,
            "expected_improvement": "20-35% reduction in false negatives",
            "difficulty": "MEDIUM - requires retraining"
        }
        
        print("📊 Expected improvement: 20-35% fewer false negatives")
        
        return strategy
    
    def data_augmentation_strategy(self):
        """
        Strategy 3: Specialized data augmentation.
        
        Implements breed-specific augmentation pipelines that address the
        particular challenges of each problematic breed category (terriers,
        nordic breeds, sighthounds).
        
        Returns:
            dict: Strategy details including augmentation pipelines for different
                  breed types, expected improvement (10-20%), and low difficulty.
        """
        print("\n🔄 STRATEGY 3: SPECIALIZED DATA AUGMENTATION")
        print("-" * 50)
        print("🎯 Objective: More data variety for problematic breeds")
        
        strategy = {
            "description": "Breed-specific augmentation based on breed type and common issues",
            "implementation": """
import torchvision.transforms as transforms
from torchvision.transforms import RandomAffine, ColorJitter, RandomHorizontalFlip

# Breed-specific augmentation pipelines
BREED_SPECIFIC_AUGMENTATION = {
    # For terriers (varied textures and sizes)
    'terriers': transforms.Compose([
        transforms.RandomRotation(15),  # Varied angles
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Varied zoom
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Fur color variations
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Position variations
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Varied focus
    ]),
    
    # For nordic breeds (huskies, malamutes)
    'nordic': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Varied body views
        transforms.ColorJitter(brightness=0.4, saturation=0.3),  # Fur variations
        transforms.RandomPerspective(distortion_scale=0.2),  # Perspective changes
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # Occlusion simulation
    ]),
    
    # For sighthounds (whippets, greyhounds - body proportions)
    'sighthounds': transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # Full body capture
        transforms.ColorJitter(contrast=0.4),  # Muscle definition
        transforms.RandomRotation(25),  # Various poses
    ])
}

def apply_breed_specific_augmentation(image, breed_name):
    \"\"\"Apply breed-specific augmentation based on breed category\"\"\"
    
    # Classify breed into category
    if breed_name in ['cairn', 'Norfolk_terrier', 'toy_terrier', 'Australian_terrier']:
        augmentation = BREED_SPECIFIC_AUGMENTATION['terriers']
    elif breed_name in ['Siberian_husky', 'malamute']:
        augmentation = BREED_SPECIFIC_AUGMENTATION['nordic'] 
    elif breed_name in ['whippet', 'Italian_greyhound']:
        augmentation = BREED_SPECIFIC_AUGMENTATION['sighthounds']
    else:
        # Standard augmentation for other breeds
        augmentation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
    
    return augmentation(image)

# Augmentation multiplier for underperforming breeds
AUGMENTATION_MULTIPLIER = {
    'Lhasa': 4,           # 4x more augmented samples
    'cairn': 4,           # 4x more augmented samples
    'Siberian_husky': 3,  # 3x more augmented samples
    'whippet': 3,         # 3x more augmented samples
    'malamute': 3,        # 3x more augmented samples
    # Normal breeds = 1x augmentation
}
            """,
            "expected_improvement": "10-20% reduction in false negatives",
            "difficulty": "LOW - does not affect current model"
        }
        
        print("📊 Expected improvement: 10-20% fewer false negatives")
        
        return strategy
    
    def focal_loss_strategy(self):
        """
        Strategy 4: Focal loss for hard-to-classify examples.
        
        Implements adaptive focal loss that automatically focuses on difficult
        examples by down-weighting easy negatives and emphasizing hard cases.
        
        Returns:
            dict: Strategy details including AdaptiveFocalLoss implementation,
                  expected improvement (25-30%), and medium difficulty level.
        """
        print("\n🧠 STRATEGY 4: FOCAL LOSS IMPLEMENTATION")
        print("-" * 50)
        print("🎯 Objective: Focus on hard-to-classify examples")
        
        strategy = {
            "description": "Use Focal Loss to give more importance to difficult examples",
            "implementation": """
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, breed_specific_gamma=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.breed_specific_gamma = breed_specific_gamma or {}
        
    def forward(self, inputs, targets, breed_names=None):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Apply breed-specific gamma values if available
        if breed_names is not None and self.breed_specific_gamma:
            gamma_values = torch.ones_like(targets, dtype=torch.float)
            for i, breed in enumerate(breed_names):
                if breed in self.breed_specific_gamma:
                    gamma_values[i] = self.breed_specific_gamma[breed]
        else:
            gamma_values = self.gamma
            
        focal_loss = self.alpha * (1 - pt) ** gamma_values * ce_loss
        return focal_loss.mean()

# Breed-specific gamma values for focus adjustment
BREED_SPECIFIC_GAMMA = {
    'Lhasa': 3.0,           # Very high focus
    'cairn': 2.8,           # High focus
    'Siberian_husky': 2.5,  # High focus
    'whippet': 2.3,         # Medium-high focus
    'malamute': 2.2,        # Medium-high focus
    # Normal breeds use default gamma
}

# Example training loop with adaptive focal loss
def train_with_adaptive_focal_loss(model, train_loader, device):
    criterion = AdaptiveFocalLoss(
        alpha=1, 
        gamma=2.0, 
        breed_specific_gamma=BREED_SPECIFIC_GAMMA
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for batch_idx, (data, targets, breed_names) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        # Use adaptive focal loss
        loss = criterion(outputs, targets, breed_names)
        loss.backward()
        optimizer.step()
            """,
            "expected_improvement": "25-30% reduction in false negatives",
            "difficulty": "MEDIUM - requires complete retraining"
        }
        
        print("📊 Expected improvement: 25-30% fewer false negatives")
        
        return strategy
        
Technical documentation in English.
print("📊 Improvement esperada: 25-30% less false negatives")
        
return strategy
    
    def ensemble_strategy(self):
        """
        Strategy 5: Ensemble methods for improved recall.
        
        Combines multiple models trained with different objectives to achieve
        better overall recall while maintaining precision.
        
        Returns:
            dict: Strategy details including RecallOptimizedEnsemble implementation,
                  expected improvement (30-40%), and high difficulty level.
        """
        print("\n📊 STRATEGY 5: ENSEMBLE METHODS")
        print("-" * 50)
        print("🎯 Objective: Combine multiple models for better recall")
        
        strategy = {
            "description": "Use an ensemble of models optimized for different aspects",
            "implementation": """
class RecallOptimizedEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
    def predict(self, x):
        predictions = []
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(x), dim=1)
                predictions.append(pred * self.weights[i])
        
        # Weighted average of predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
    
    def predict_with_recall_boost(self, x, breed_name, recall_boost_factor=1.2):
        base_prediction = self.predict(x)
        
        # Boost for breeds with recall problems
        if breed_name in ['Lhasa', 'cairn', 'Siberian_husky', 'whippet']:
            # Increase probability for the correct class
            class_idx = get_breed_index(breed_name)
            base_prediction[:, class_idx] *= recall_boost_factor
            
            # Renormalize probabilities
            base_prediction = torch.softmax(base_prediction, dim=1)
        
        return base_prediction

# Create specialized ensemble
def create_recall_optimized_ensemble():
    # Model 1: Optimized for general precision
    model1 = load_model('best_model_fold_0.pth')
    
    # Model 2: Trained with focal loss
    model2 = load_model('focal_loss_model.pth')
    
    # Model 3: Trained with weighted loss
    model3 = load_model('weighted_model.pth')
    
    # Higher weight for focal loss model (better recall)
    ensemble_weights = [0.3, 0.4, 0.3]
    
    return RecallOptimizedEnsemble([model1, model2, model3], ensemble_weights)

# Example inference with ensemble
ensemble = create_recall_optimized_ensemble()
prediction = ensemble.predict_with_recall_boost(image, breed_name)
            """,
            "expected_improvement": "30-40% reduction in false negatives",
            "difficulty": "HIGH - requires multiple trained models"
        }
        
        print("📊 Expected improvement: 30-40% fewer false negatives")
        
        return strategy
    
    def generate_implementation_roadmap(self):
        """
        Generate a phased implementation roadmap.
        
        Creates a structured plan for implementing false negative corrections
        across four phases, from immediate quick fixes to long-term solutions.
        
        Returns:
            dict: Roadmap with four phases, each containing timeframe,
                  actions, expected improvement, and effort level.
        """
        print("\n" + "=" * 70)
        print("🗺️ IMPLEMENTATION ROADMAP - FALSE NEGATIVE CORRECTION")
        print("=" * 70)
        
        roadmap = {
            "Phase_1_Immediate": {
                "timeframe": "1-2 days",
                "actions": [
                    "✅ Implement per-class threshold adjustment",
                    "✅ Apply lower thresholds to critical breeds",
                    "✅ Immediate testing on problematic breeds"
                ],
                "expected_improvement": "15-25%",
                "effort": "LOW"
            },
            "Phase_2_Short_term": {
                "timeframe": "1 week", 
                "actions": [
                    "🔄 Implement specialized augmentation",
                    "📸 Generate more data for critical breeds",
                    "🧪 Test with new augmented data"
                ],
                "expected_improvement": "25-35%",
                "effort": "MEDIUM"
            },
            "Phase_3_Medium_term": {
                "timeframe": "2-3 weeks",
                "actions": [
                    "🎯 Implement Weighted/Focal Loss",
                    "🔄 Retrain model with new loss functions",
                    "📊 Complete model validation"
                ],
                "expected_improvement": "35-50%",
                "effort": "HIGH"
            },
            "Phase_4_Long_term": {
                "timeframe": "1 month",
                "actions": [
                    "📊 Implement ensemble methods",
                    "🔧 Complete pipeline optimization",
                    "🚀 Production deployment"
                ],
                "expected_improvement": "50-60%",
                "effort": "VERY HIGH"
            }
        }
        
        for phase, details in roadmap.items():
            print(f"\n🎯 {phase.replace('_', ' ').upper()}")
            print(f"   ⏱️  Timeframe: {details['timeframe']}")
            print(f"   📈 Expected improvement: {details['expected_improvement']}")
            print(f"   💪 Effort level: {details['effort']}")
            print("   📋 Actions:")
            for action in details['actions']:
                print(f"      {action}")
        
        return roadmap
    
        def create_quick_fix_script(self):
        """
        Create a quick-fix script for immediate deployment.
        
        Generates a ready-to-use Python script implementing adaptive threshold
        classification that can be deployed immediately without retraining.
        
        Returns:
            str: Python script code for ThresholdOptimizedClassifier.
        """
        print("\n" + "=" * 60)
        print("⚡ QUICK FIX SCRIPT - READY TO USE")
        print("=" * 60)
        
        quick_fix_code = '''
# Quick False Negative Fix - Immediate Deployment Script
# File: quick_false_negative_fix.py

import torch
import numpy as np

class ThresholdOptimizedClassifier:
    """
    Classifier with adaptive per-breed thresholds for reduced false negatives.
    
    This classifier wraps a base model and applies breed-specific thresholds
    that are lower for breeds historically prone to false negatives.
    
    Args:
        base_model: The pretrained classification model.
        breed_thresholds (dict, optional): Custom thresholds per breed.
    """
    
    def __init__(self, base_model, breed_thresholds=None):
        self.base_model = base_model
        self.breed_thresholds = breed_thresholds or {
            'Lhasa': 0.35,           # Very low (was 46% FN)
            'cairn': 0.40,           # Low (was 41% FN)
            'Siberian_husky': 0.45,  # Low-medium (was 38% FN)
            'whippet': 0.45,         # Low-medium (was 36% FN)
            'malamute': 0.50,        # Medium (was 35% FN)
            'Australian_terrier': 0.50,
            'Norfolk_terrier': 0.50,
            'toy_terrier': 0.55,
            'Italian_greyhound': 0.55,
        }
        self.default_threshold = 0.60
        
    def predict_with_adaptive_thresholds(self, image, breed_names):
        """
        Predict with breed-specific adaptive thresholds.
        
        Args:
            image: Input image tensor.
            breed_names: List of breed names corresponding to class indices.
            
        Returns:
            list: Sorted list of prediction dictionaries.
        """
        # Get predictions from base model
        with torch.no_grad():
            logits = self.base_model(image)
            probabilities = torch.softmax(logits, dim=1)
        
        results = []
        
        for i, breed in enumerate(breed_names):
            prob_score = probabilities[0][i].item()
            threshold = self.breed_thresholds.get(breed, self.default_threshold)
            
            # Apply adaptive threshold
            is_predicted = prob_score >= threshold
            
            results.append({
                'breed': breed,
                'probability': prob_score,
                'threshold_used': threshold,
                'predicted': is_predicted,
                'improvement': 'OPTIMIZED' if breed in self.breed_thresholds else 'STANDARD'
            })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)

# USAGE:
# 1. Load your current model
# model = torch.load('best_model_fold_0.pth')
#    
# 2. Create optimized classifier
# optimized_classifier = ThresholdOptimizedClassifier(model)
#    
# 3. Use with images
# results = optimized_classifier.predict_with_adaptive_thresholds(image, breed_names)
'''
        
        # Save the quick fix script to file
        with open('quick_false_negative_fix.py', 'w') as f:
            f.write(quick_fix_code)
        
        print("💾 Script saved as: quick_false_negative_fix.py")
        print("⚡ READY for IMMEDIATE use!")
        
        return quick_fix_code


def main():
    """
    Main function to generate and display the false negative correction plan.
    
    Creates a FalseNegativeCorrector instance and generates all correction
    strategies, implementation roadmap, and quick-fix scripts.
    
    Returns:
        dict: Dictionary containing all strategies, roadmap, and quick-fix status.
    """
    corrector = FalseNegativeCorrector()
    
    # Generate correction strategies
    strategies = corrector.generate_correction_plan()
    
    # Generate implementation roadmap
    roadmap = corrector.generate_implementation_roadmap()
    
    # Create ready-to-use quick fix script
    corrector.create_quick_fix_script()
    
    print("\n" + "=" * 70)
    print("🌟 FALSE NEGATIVE CORRECTION - SUMMARY")
    print("=" * 70)
    print("\n📋 RECOMMENDED NEXT STEPS:")
    print(" 1. ⚡ Use 'quick_false_negative_fix.py' IMMEDIATELY")
    print(" 2. 🧪 Test on problematic breeds (Lhasa, cairn, husky, whippet)")
    print(" 3. 📊 Measure recall improvement")
    print(" 4. 🔄 Proceed with Phase 2 if results are satisfactory")
    
    return {
        'strategies': strategies,
        'roadmap': roadmap,
        'quick_fix_ready': True
    }


if __name__ == "__main__":
    main()