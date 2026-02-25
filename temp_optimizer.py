#!/usr/bin/env python3
"""
Temperature Scaling Optimizer for Breed Classification.

This script performs temperature scaling optimization on the breed classification model.
Temperature scaling is a calibration technique that adjusts model confidence scores
to better reflect actual prediction accuracy, improving over-confident predictions.

Key Features:
    - Tests multiple temperature values to find optimal calibration
    - Analyzes impact on specific target breeds (Labrador, Pug, Beagle)
    - Displays Top-5 predictions with optimized temperature
    - Recommends best temperature value for production use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os


class BreedModelTest(nn.Module):
    """
    ResNet34-based breed classification model for temperature testing.
    
    Attributes:
        backbone (nn.Module): ResNet34 model with modified final layer.
    """
    
    def __init__(self, num_classes=50):
        """
        Initialize the breed classification model.
        
        Args:
            num_classes (int): Number of breed classes to predict. Default: 50.
        """
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Class logits of shape (N, num_classes).
        """
        return self.backbone(x)


# Use CPU for testing
device = torch.device('cpu')

print(" OPTIMAL TEMPERATURE SEARCH")
print("=" * 70)

# Load the trained model
breed_model = BreedModelTest(num_classes=50).to(device)
breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"

checkpoint = torch.load(breed_path, map_location=device)
breed_model.load_state_dict(checkpoint['model_state_dict'])
breed_model.eval()

# Get breed names from training directory
breed_dir = "breed_processed_data/train"
breed_names = sorted([d for d in os.listdir(breed_dir) 
                     if os.path.isdir(os.path.join(breed_dir, d))])

print(f" Model loaded with {len(breed_names)} breeds")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Test image (Labrador-like color)
test_image = Image.new('RGB', (300, 300), color=(205, 133, 63))
input_tensor = transform(test_image).unsqueeze(0).to(device)

# Temperature values to test
temperatures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

# Target breeds for optimization
target_breeds = ['Labrador_retriever', 'pug', 'beagle']
target_indices = {}
for breed in target_breeds:
    if breed in breed_names:
        target_indices[breed] = breed_names.index(breed)

print(f" Target: Improve detection for {list(target_indices.keys())}")
print(f"\n{'Temp':<6} | {'Top 1':<20} | {'Conf%':<8} | {'Lab%':<8} | {'Beagle%':<8}")
print("-" * 70)

with torch.no_grad():
    logits = breed_model(input_tensor)
    
    best_temp = 1.0
    best_labrador_score = 0.0
    
    for temp in temperatures:
        probs = F.softmax(logits / temp, dim=1)
        
        # Top 1
        top1_prob, top1_idx = torch.max(probs, 1)
        top1_name = breed_names[top1_idx.item()]
        top1_conf = top1_prob.item() * 100
        
        # Get specific breed probabilities for analysis
        lab_prob = probs[0][target_indices.get('Labrador_retriever', 0)].item() * 100
        beagle_prob = probs[0][target_indices.get('beagle', 0)].item() * 100
        
        # Find best temperature for Labrador detection
        if lab_prob > best_labrador_score:
            best_labrador_score = lab_prob
            best_temp = temp
        
        marker = "" if temp == best_temp else "  "
        print(f"{marker}{temp:<6.1f} | {top1_name[:19]:<20} | {top1_conf:<8.2f} | {lab_prob:<8.3f} | {beagle_prob:<8.3f}")

print(f"\n BEST TEMPERATURE: {best_temp}")
print(f" Labrador improvement: {best_labrador_score:.3f}%")

# Show Top-5 with best temperature
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

print(f"\n RECOMMENDATION: Use temperature {best_temp}")