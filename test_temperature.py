# !/usr/bin/env python3
"""
Test of the Temperature Scaling aplicado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def test_temperature_scaling():
    print("🌡️ PRUEBA DE TEMPERATURE SCALING")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # load model
    breed_model = BreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Get names of breeds
    breed_dir = "breed_processed_data/train"
    breed_names = sorted([d for d in os.listdir(breed_dir) 
                         if os.path.isdir(os.path.join(breed_dir, d))])
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Implementation note.
    test_image = Image.new('RGB', (300, 300), color=(139, 69, 19))
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    # Probar diferentes temperaturas
    temperatures = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    print(f"🔬 Analizando con diferentes temperaturas...")
    print(f"{'Temp':<6} | {'Top 1':<20} | {'Conf%':<8} | {'Top 2':<20} | {'Conf%':<8}")
    print("-" * 80)
    
    with torch.no_grad():
        # Get logits a sola vez
        logits = breed_model(input_tensor)
        
        for temp in temperatures:
            # Apply temperatura
            probs = F.softmax(logits / temp, dim=1)
            
            # Top 2 predictions
            top2_probs, top2_indices = torch.topk(probs, 2, dim=1)
            
            top1_name = breed_names[top2_indices[0][0].item()]
            top1_prob = top2_probs[0][0].item() * 100
            
            top2_name = breed_names[top2_indices[0][1].item()]  
            top2_prob = top2_probs[0][1].item() * 100
            
            print(f"{temp:<6.1f} | {top1_name:<20} | {top1_prob:<8.2f} | {top2_name:<20} | {top2_prob:<8.2f}")
            
            # Implementation note.
            target_indices = {
                'pug': breed_names.index('pug') if 'pug' in breed_names else -1,
                'Labrador_retriever': breed_names.index('Labrador_retriever') if 'Labrador_retriever' in breed_names else -1,
                'Norwegian_elkhound': breed_names.index('Norwegian_elkhound') if 'Norwegian_elkhound' in breed_names else -1
            }
            
            if temp == 2.5:  # Nuestra temperatura elegida
                print(f"\n🎯 DETALLES CON TEMPERATURA {temp}:")
                for breed, idx in target_indices.items():
                    if idx >= 0:
                        prob = probs[0][idx].item() * 100
                        print(f"   {breed:<20}: {prob:6.3f}%")
    
    print("\n" + "=" * 60)
    print("🌡️ Temperature Scaling explicación:")
    print("   • Temp = 1.0: Predicciones originales (muy extremas)")
    print("   • Temp > 1.0: Predicciones más suaves y distribuidas")
    print("   • Temp = 2.5: Nuestro valor elegido (balance)")
    print("   • Temp alta: Muy distribuido (menos confianza)")

if __name__ == "__main__":
    test_temperature_scaling()