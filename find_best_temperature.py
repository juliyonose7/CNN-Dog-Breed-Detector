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
    print("🌡️ BÚSQUEDA DE TEMPERATURA ÓPTIMA")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    # load model
    breed_model = SimpleBreedModel(num_classes=50).to(device)
    breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    
    checkpoint = torch.load(breed_path, map_location=device)
    breed_model.load_state_dict(checkpoint['model_state_dict'])
    breed_model.eval()
    
    # Obtener names de breeds
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
    
    # Crear image de prueba (Labrador color)
    test_image = Image.new('RGB', (300, 300), color=(205, 133, 63))  # Sandy brown
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    # Implementation note.
    temperatures = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    
    # Breeds objetivo that queremos mejorar
    target_breeds = ['Labrador_retriever', 'pug', 'beagle']
    target_indices = {}
    for breed in target_breeds:
        if breed in breed_names:
            target_indices[breed] = breed_names.index(breed)
    
    print(f"🎯 Objetivo: Mejorar detección de {list(target_indices.keys())}")
    print(f"🔬 Probando temperaturas: {temperatures}")
    print("\n" + "=" * 70)
    
    with torch.no_grad():
        logits = breed_model(input_tensor)
        
        print(f"{'Temp':<6} | {'Top 1':<20} | {'Conf%':<8} | {'Lab%':<8} | {'Pug%':<8} | {'Beagle%':<8}")
        print("-" * 75)
        
        best_temp = 1.0
        best_labrador_score = 0.0
        
        for temp in temperatures:
            probs = F.softmax(logits / temp, dim=1)
            
            # Top 1
            top1_prob, top1_idx = torch.max(probs, 1)
            top1_name = breed_names[top1_idx.item()]
            top1_conf = top1_prob.item() * 100
            
            # Implementation note.
            lab_prob = probs[0][target_indices['Labrador_retriever']].item() * 100 if 'Labrador_retriever' in target_indices else 0
            pug_prob = probs[0][target_indices['pug']].item() * 100 if 'pug' in target_indices else 0
            beagle_prob = probs[0][target_indices['beagle']].item() * 100 if 'beagle' in target_indices else 0
            
            # Buscar best temperatura for Labrador
            if lab_prob > best_labrador_score:
                best_labrador_score = lab_prob
                best_temp = temp
            
            marker = "🔥" if temp == best_temp else "  "
            print(f"{marker}{temp:<6.1f} | {top1_name[:19]:<20} | {top1_conf:<8.2f} | {lab_prob:<8.3f} | {pug_prob:<8.3f} | {beagle_prob:<8.3f}")
        
        print("\n" + "=" * 70)
        print(f"🏆 MEJOR TEMPERATURA PARA LABRADOR: {best_temp}")
        print(f"📈 Mejora en Labrador: {best_labrador_score:.3f}%")
        
        # Mostrar top 5 with la best temperatura
        print(f"\n🔥 TOP 5 CON TEMPERATURA {best_temp}:")
        probs_best = F.softmax(logits / best_temp, dim=1)
        top5_probs, top5_indices = torch.topk(probs_best, 5, dim=1)
        
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item() * 100
            breed = breed_names[idx]
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            special = "🎯" if breed in target_breeds else "  "
            print(f"{special} {medal} {breed:<25} {prob:>8.3f}%")
    
    return best_temp

if __name__ == "__main__":
    best_temperature = main()
    print(f"\n✅ Usar temperatura: {best_temperature}")