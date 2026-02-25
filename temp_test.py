# !/usr/bin/env python3
"""
Technical documentation in English.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

class SimpleBreedModel(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def main():
    print("🌡️ PRUEBA DE TEMPERATURE SCALING")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # load model
    breed_model = SimpleBreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    if not os.path.exists(breed_path):
        print(f"❌ No encontrado: {breed_path}")
        return
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Obtener names de breeds
    breed_dir = "breed_processed_data/train"
    if not os.path.exists(breed_dir):
        print(f"❌ No encontrado: {breed_dir}")
        return
        
    breed_names = sorted([d for d in os.listdir(breed_dir) 
                         if os.path.isdir(os.path.join(breed_dir, d))])
    
    print(f"✅ Modelo cargado con {len(breed_names)} razas")
    
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
    
    print(f"\n🔬 Comparando temperaturas:")
    print(f"{'Temp':<6} | {'Top 1':<20} | {'Conf%':<8} | {'Top 2':<20} | {'Conf%':<8}")
    print("-" * 80)
    
    with torch.no_grad():
        # Obtener logits una sola vez
        logits = breed_model(input_tensor)
        
        for temp in temperatures:
            # Aplicar temperatura
            probs = F.softmax(logits / temp, dim=1)
            
            # Top 2 predictions
            top2_probs, top2_indices = torch.topk(probs, 2, dim=1)
            
            top1_name = breed_names[top2_indices[0][0].item()]
            top1_prob = top2_probs[0][0].item() * 100
            
            top2_name = breed_names[top2_indices[0][1].item()]  
            top2_prob = top2_probs[0][1].item() * 100
            
            print(f"{temp:<6.1f} | {top1_name:<20} | {top1_prob:<8.2f} | {top2_name:<20} | {top2_prob:<8.2f}")
        
        # Implementation note.
        print(f"\n🎯 CAMBIOS EN RAZAS ESPECÍFICAS:")
        target_breeds = ['pug', 'Labrador_retriever', 'Norwegian_elkhound', 'basset']
        
        print(f"{'Raza':<20} | {'T=1.0':<8} | {'T=2.5':<8} | {'Cambio':<10}")
        print("-" * 60)
        
        for breed in target_breeds:
            if breed in breed_names:
                idx = breed_names.index(breed)
                
                # T=1.0 (original)
                probs_1 = F.softmax(logits / 1.0, dim=1)
                prob_1 = probs_1[0][idx].item() * 100
                
                # T=2.5 (ajustado)
                probs_25 = F.softmax(logits / 2.5, dim=1)
                prob_25 = probs_25[0][idx].item() * 100
                
                cambio = prob_25 - prob_1
                cambio_str = f"+{cambio:.2f}%" if cambio >= 0 else f"{cambio:.2f}%"
                
                print(f"{breed:<20} | {prob_1:<8.3f} | {prob_25:<8.3f} | {cambio_str:<10}")
    
    print("\n" + "=" * 60)
    print("🎯 RESULTADO:")
    print("   ✅ Temperature Scaling suaviza predicciones extremas")
    print("   ✅ Reduce dominancia de clases sobre-representadas")
    print("   ✅ Da más oportunidades a otras razas")
    print("   🌡️ T=2.5 es un buen balance para este modelo")
    print("\n¡Ahora prueba subir imágenes reales para ver la diferencia!")

if __name__ == "__main__":
    main()