#!/usr/bin/env python3
"""
Temperature Scaling Analysis Module.

This script analyzes the effect of temperature scaling on model predictions.
Temperature scaling is a post-hoc calibration technique that adjusts the
confidence of neural network predictions without retraining.

Key Concepts:
    - Temperature = 1.0: Original predictions (often overconfident)
    - Temperature > 1.0: Softer, more distributed predictions
    - Higher temperatures reduce confidence peaks

Usage:
    python test_temperature.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os


class BreedModel(nn.Module):
    """
    Breed classification model based on ResNet34.
    
    Attributes:
        backbone (nn.Module): ResNet34 feature extractor with custom FC layer.
    """
    
    def __init__(self, num_classes: int = 50):
        """
        Initialize the breed classification model.
        
        Args:
            num_classes (int): Number of breed classes. Default: 50.
        """
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Logit outputs.
        """
        return self.backbone(x)

def test_temperature_scaling():
    """
    Test the effect of temperature scaling on breed predictions.
    
    Loads the trained breed model and analyzes how different temperature
    values affect the probability distribution of predictions.
    """
    print(" TEMPERATURE SCALING TEST")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Load model
    breed_model = BreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Get breed names
    breed_dir = "breed_processed_data/train"
    breed_names = sorted([d for d in os.listdir(breed_dir) 
                         if os.path.isdir(os.path.join(breed_dir, d))])
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create synthetic test image (brown rectangle simulating a dog)
    test_image = Image.new('RGB', (300, 300), color=(139, 69, 19))
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    # Test different temperatures
    temperatures = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    print(f" Analyzing with different temperatures...")
    print(f"{'Temp':<6} | {'Top 1':<20} | {'Conf%':<8} | {'Top 2':<20} | {'Conf%':<8}")
    print("-" * 80)
    
    with torch.no_grad():
        # Get logits once
        logits = breed_model(input_tensor)
        
        for temp in temperatures:
            # Apply temperature
            probs = F.softmax(logits / temp, dim=1)
            
            # Top 2 predictions
            top2_probs, top2_indices = torch.topk(probs, 2, dim=1)
            
            top1_name = breed_names[top2_indices[0][0].item()]
            top1_prob = top2_probs[0][0].item() * 100
            
            top2_name = breed_names[top2_indices[0][1].item()]  
            top2_prob = top2_probs[0][1].item() * 100
            
            print(f"{temp:<6.1f} | {top1_name:<20} | {top1_prob:<8.2f} | {top2_name:<20} | {top2_prob:<8.2f}")
            
            # Show specific breed probabilities
            target_indices = {
                'pug': breed_names.index('pug') if 'pug' in breed_names else -1,
                'Labrador_retriever': breed_names.index('Labrador_retriever') if 'Labrador_retriever' in breed_names else -1,
                'Norwegian_elkhound': breed_names.index('Norwegian_elkhound') if 'Norwegian_elkhound' in breed_names else -1
            }
            
            if temp == 2.5:  # Our chosen temperature
                print(f"\n DETAILS AT TEMPERATURE {temp}:")
                for breed, idx in target_indices.items():
                    if idx >= 0:
                        prob = probs[0][idx].item() * 100
                        print(f"   {breed:<20}: {prob:6.3f}%")
    
    print("\n" + "=" * 60)
    print(" Temperature Scaling explanation:")
    print("   • Temp = 1.0: Original predictions (very extreme)")
    print("   • Temp > 1.0: Softer, more distributed predictions")
    print("   • Temp = 2.5: Our chosen value (balanced)")
    print("   • High Temp: Very distributed (less confidence)")

if __name__ == "__main__":
    test_temperature_scaling()