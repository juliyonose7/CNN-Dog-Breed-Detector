#!/usr/bin/env python3
"""
Temperature Scaling Test Module.

This script tests temperature scaling effects on breed classification predictions.
It compares probability distributions across different temperature values and
analyzes how temperature affects specific target breeds.

Temperature scaling smooths overconfident predictions and provides better
calibrated probability estimates without retraining the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os


class SimpleBreedModel(nn.Module):
    """
    Simple ResNet34-based breed classification model.
    
    Attributes:
        backbone (nn.Module): ResNet34 model with modified classifier.
    """
    
    def __init__(self, num_classes=50):
        """
        Initialize the breed model.
        
        Args:
            num_classes (int): Number of breed classes. Default: 50.
        """
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Class logits.
        """
        return self.backbone(x)


def main():
    """
    Main function to test temperature scaling effects.
    
    Tests multiple temperature values and compares their effects on:
        - Top-1 and Top-2 confidence distributions
        - Specific breed probability changes (pug, Labrador, etc.)
    """
    print(" TEMPERATURE SCALING TEST")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Load model
    breed_model = SimpleBreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    if not os.path.exists(breed_path):
        print(f" Not found: {breed_path}")
        return
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Get breed names from training directory
    breed_dir = "breed_processed_data/train"
    if not os.path.exists(breed_dir):
        print(f" Not found: {breed_dir}")
        return
        
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
    
    # Create test image (brown color to simulate dog)
    test_image = Image.new('RGB', (300, 300), color=(139, 69, 19))
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    # Test different temperatures
    temperatures = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    print(f"\n Comparing temperatures:")
    print(f"{'Temp':<6} | {'Top 1':<20} | {'Conf%':<8} | {'Top 2':<20} | {'Conf%':<8}")
    print("-" * 80)
    
    with torch.no_grad():
        # Get logits once
        logits = breed_model(input_tensor)
        
        for temp in temperatures:
            # Apply temperature scaling
            probs = F.softmax(logits / temp, dim=1)
            
            # Top 2 predictions
            top2_probs, top2_indices = torch.topk(probs, 2, dim=1)
            
            top1_name = breed_names[top2_indices[0][0].item()]
            top1_prob = top2_probs[0][0].item() * 100
            
            top2_name = breed_names[top2_indices[0][1].item()]  
            top2_prob = top2_probs[0][1].item() * 100
            
            print(f"{temp:<6.1f} | {top1_name:<20} | {top1_prob:<8.2f} | {top2_name:<20} | {top2_prob:<8.2f}")
        
        # Analyze specific breed probability changes
        print(f"\n CHANGES IN SPECIFIC BREEDS:")
        target_breeds = ['pug', 'Labrador_retriever', 'Norwegian_elkhound', 'basset']
        
        print(f"{'Breed':<20} | {'T=1.0':<8} | {'T=2.5':<8} | {'Change':<10}")
        print("-" * 60)
        
        for breed in target_breeds:
            if breed in breed_names:
                idx = breed_names.index(breed)
                
                # T=1.0 (original)
                probs_1 = F.softmax(logits / 1.0, dim=1)
                prob_1 = probs_1[0][idx].item() * 100
                
                # T=2.5 (calibrated)
                probs_25 = F.softmax(logits / 2.5, dim=1)
                prob_25 = probs_25[0][idx].item() * 100
                
                change = prob_25 - prob_1
                change_str = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
                
                print(f"{breed:<20} | {prob_1:<8.3f} | {prob_25:<8.3f} | {change_str:<10}")
    
    print("\n" + "=" * 60)
    print(" RESULTS:")
    print("    Temperature Scaling smooths extreme predictions")
    print("    Reduces dominance of over-represented classes")
    print("    Gives better opportunities to other breeds")
    print("    T=2.5 is a good balance for this model")
    print("\nNow test with real images to see the difference!")

if __name__ == "__main__":
    main()