# !/usr/bin/env python3
"""
Technical documentation in English.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class BreedModel(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def test_specific_breeds():
    print("🧪 PRUEBA ESPECÍFICA DE RAZAS PROBLEMÁTICAS")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # load model
    breed_model = BreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Get names of breeds of the directory
    breed_dir = "breed_processed_data/train"
    breed_names = sorted([d for d in os.listdir(breed_dir) 
                         if os.path.isdir(os.path.join(breed_dir, d))])
    
    print(f"📋 {len(breed_names)} razas cargadas:")
    for i, breed in enumerate(breed_names):
        marker = "🎯" if breed in ['pug', 'Labrador_retriever', 'Norwegian_elkhound'] else "  "
        print(f"{marker} {i:2d}: {breed}")
    
    # Implementation note.
    target_breeds = ['pug', 'Labrador_retriever', 'Norwegian_elkhound']
    breed_indices = {}
    for target in target_breeds:
        if target in breed_names:
            breed_indices[target] = breed_names.index(target)
            print(f"\n🎯 {target} -> Índice {breed_indices[target]}")
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create image of test
    test_image = Image.new('RGB', (300, 300), color=(139, 69, 19))  # Implementation note.
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    print(f"\n🔬 Analizando predicciones...")
    
    with torch.no_grad():
        output = breed_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Top 10 predictions
        top_probs, top_indices = torch.topk(probabilities, 10, dim=1)
        
        print(f"\n📊 TOP 10 PREDICCIONES:")
        for i in range(10):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            breed_name = breed_names[idx] if idx < len(breed_names) else f"UNKNOWN_{idx}"
            
            marker = "🔴" if breed_name in target_breeds else "  "
            print(f"{marker} {i+1:2d}. {breed_name:<25} -> {prob:.4f} ({prob*100:.2f}%)")
        
        # Implementation note.
        print(f"\n🎯 PROBABILIDADES ESPECÍFICAS:")
        for breed, idx in breed_indices.items():
            prob = probabilities[0][idx].item()
            print(f"   {breed:<20} (idx {idx:2d}): {prob:.6f} ({prob*100:.3f}%)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_specific_breeds()