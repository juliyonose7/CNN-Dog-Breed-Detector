# !/usr/bin/env python3
"""
Breed Prediction Debugging Module.

This module provides tools for testing and debugging breed classification
predictions. It loads the trained model and performs inference on test
images while displaying detailed probability distributions to help identify
prediction issues.

Key Features:
    - Model architecture verification
    - Breed class loading from directory structure
    - Test image generation for quick debugging
    - Top-K prediction analysis with probabilities
    - Specific breed probability inspection

Usage:
    python debug_predictions.py

Author: System IA
Date: 2024
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


class BreedModel(nn.Module):
    """
    Breed classification model using ResNet34 backbone.

    This model uses a pretrained ResNet34 architecture with a custom
    fully connected layer for multi-class breed classification.

    Attributes:
        backbone (nn.Module): ResNet34 feature extractor with custom FC layer.

    Args:
        num_classes (int): Number of breed classes to classify. Defaults to 50.
    """

    def __init__(self, num_classes=50):
        """
        Initialize the breed classification model.

        Args:
            num_classes (int): Number of output classes for breed classification.
        """
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)


def test_specific_breeds():
    """
    Test predictions for specific problematic breeds.

    This function performs a diagnostic test by:
    1. Loading the breed classification model
    2. Creating a synthetic test image
    3. Analyzing the prediction distribution
    4. Reporting probabilities for specific breeds of interest

    The function is useful for debugging classification issues with
    particular breeds that may have low accuracy or confusion with
    other breeds.

    Returns:
        None: Prints diagnostic information to stdout.

    Note:
        Uses a brown synthetic test image (139, 69, 19 RGB) which may
        not produce meaningful predictions but helps verify model loading.
    """
    print("🧪 SPECIFIC BREED TESTING FOR PROBLEMATIC CLASSES")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Load model
    breed_model = BreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Get breed names from directory structure
    breed_dir = "breed_processed_data/train"
    breed_names = sorted([d for d in os.listdir(breed_dir) 
                         if os.path.isdir(os.path.join(breed_dir, d))])
    
    print(f"📋 {len(breed_names)} breeds loaded:")
    for i, breed in enumerate(breed_names):
        marker = "🎯" if breed in ['pug', 'Labrador_retriever', 'Norwegian_elkhound'] else "  "
        print(f"{marker} {i:2d}: {breed}")
    
    # Find indices for target breeds
    target_breeds = ['pug', 'Labrador_retriever', 'Norwegian_elkhound']
    breed_indices = {}
    for target in target_breeds:
        if target in breed_names:
            breed_indices[target] = breed_names.index(target)
            print(f"\n🎯 {target} -> Index {breed_indices[target]}")
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create synthetic test image (brown color)
    test_image = Image.new('RGB', (300, 300), color=(139, 69, 19))
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    print(f"\n🔬 Analyzing predictions...")
    
    with torch.no_grad():
        output = breed_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Top 10 predictions
        top_probs, top_indices = torch.topk(probabilities, 10, dim=1)
        
        print(f"\n📊 TOP 10 PREDICTIONS:")
        for i in range(10):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            breed_name = breed_names[idx] if idx < len(breed_names) else f"UNKNOWN_{idx}"
            
            marker = "🔴" if breed_name in target_breeds else "  "
            print(f"{marker} {i+1:2d}. {breed_name:<25} -> {prob:.4f} ({prob*100:.2f}%)")
        
        # Specific breed probabilities
        print(f"\n🎯 SPECIFIC BREED PROBABILITIES:")
        for breed, idx in breed_indices.items():
            prob = probabilities[0][idx].item()
            print(f"   {breed:<20} (idx {idx:2d}): {prob:.6f} ({prob*100:.3f}%)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_specific_breeds()