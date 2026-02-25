#!/usr/bin/env python3
"""
Optimal Temperature Scaling Finder Module
=========================================

This module searches for the optimal temperature scaling parameter to improve
model calibration and prediction confidence distribution for a breed classifier.

Temperature scaling is a post-hoc calibration method where logits are divided
by a temperature value T before applying softmax. Higher temperatures produce
softer (more uniform) probability distributions, while lower temperatures
produce sharper (more confident) distributions.

Features:
    - Tests multiple temperature values across a range
    - Evaluates impact on target breed detection
    - Displays comparative analysis of probabilities
    - Identifies optimal temperature for specific breed improvement

Usage:
    Run this script to find the best temperature for softening predictions
    and improving detection of underperforming breeds.

Author: AI System
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

class SimpleBreedModel(nn.Module):
    """
    Simple breed classification model based on ResNet34.
    
    A lightweight breed classifier using ResNet34 backbone with a modified
    final fully connected layer for multi-class breed classification.
    
    Args:
        num_classes (int): Number of breed classes to classify (default: 50).
    
    Architecture:
        - Backbone: ResNet34 (without pretrained weights for loading custom)
        - Output: Linear layer mapping to num_classes
    """
    
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch, 3, H, W).
        
        Returns:
            torch.Tensor: Raw logits of shape (batch, num_classes).
        """
        return self.backbone(x)

def main():
    """
    Main function to search for optimal temperature scaling parameter.
    
    Loads a trained breed model and tests various temperature values to find
    the one that best improves detection of target breeds (Labrador, pug, beagle).
    
    The function:
        1. Loads the trained breed classification model
        2. Creates a test image to evaluate temperature effects
        3. Tests temperatures from 1.0 to 10.0
        4. Reports which temperature maximizes target breed probabilities
        5. Displays top-5 predictions with the optimal temperature
    
    Returns:
        float: The optimal temperature value for the target breeds.
    """
    print(" OPTIMAL TEMPERATURE SEARCH")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    # Load trained breed model
    breed_model = SimpleBreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Get breed names from training directory
    breed_dir = "breed_processed_data/train"
    breed_names = sorted([d for d in os.listdir(breed_dir) 
                         if os.path.isdir(os.path.join(breed_dir, d))])
    
    # Image preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create synthetic test image (Labrador-like color)
    test_image = Image.new('RGB', (300, 300), color=(205, 133, 63))  # Sandy brown
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    # Temperature values to test
    temperatures = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    
    # Target breeds we want to improve detection for
    target_breeds = ['Labrador_retriever', 'pug', 'beagle']
    target_indices = {}
    for breed in target_breeds:
        if breed in breed_names:
            target_indices[breed] = breed_names.index(breed)
    
    print(f" Objective: Improve detection of {list(target_indices.keys())}")
    print(f" Testing temperatures: {temperatures}")
    print("\n" + "=" * 70)
    
    with torch.no_grad():
        logits = breed_model(input_tensor)
        
        print(f"{'Temp':<6} | {'Top 1':<20} | {'Conf%':<8} | {'Lab%':<8} | {'Pug%':<8} | {'Beagle%':<8}")
        print("-" * 75)
        
        best_temp = 1.0
        best_labrador_score = 0.0
        
        for temp in temperatures:
            probs = F.softmax(logits / temp, dim=1)
            
            # Get top-1 prediction
            top1_prob, top1_idx = torch.max(probs, 1)
            top1_name = breed_names[top1_idx.item()]
            top1_conf = top1_prob.item() * 100
            
            # Get probabilities for target breeds
            lab_prob = probs[0][target_indices['Labrador_retriever']].item() * 100 if 'Labrador_retriever' in target_indices else 0
            pug_prob = probs[0][target_indices['pug']].item() * 100 if 'pug' in target_indices else 0
            beagle_prob = probs[0][target_indices['beagle']].item() * 100 if 'beagle' in target_indices else 0
            
            # Track best temperature for Labrador
            if lab_prob > best_labrador_score:
                best_labrador_score = lab_prob
                best_temp = temp
            
            marker = "" if temp == best_temp else "  "
            print(f"{marker}{temp:<6.1f} | {top1_name[:19]:<20} | {top1_conf:<8.2f} | {lab_prob:<8.3f} | {pug_prob:<8.3f} | {beagle_prob:<8.3f}")
        
        print("\n" + "=" * 70)
        print(f" BEST TEMPERATURE FOR LABRADOR: {best_temp}")
        print(f" Labrador probability improvement: {best_labrador_score:.3f}%")
        
        # Display top-5 predictions with optimal temperature
        print(f"\n TOP 5 WITH TEMPERATURE {best_temp}:")
        probs_best = F.softmax(logits / best_temp, dim=1)
        top5_probs, top5_indices = torch.topk(probs_best, 5, dim=1)
        
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item() * 100
            breed = breed_names[idx]
            medal = ["", "", "", "4", "5"][i]
            special = "" if breed in target_breeds else "  "
            print(f"{special} {medal} {breed:<25} {prob:>8.3f}%")
    
    return best_temp

if __name__ == "__main__":
    best_temperature = main()
    print(f"\n Recommended temperature: {best_temperature}")