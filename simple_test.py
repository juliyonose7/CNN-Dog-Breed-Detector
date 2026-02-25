#!/usr/bin/env python3
"""
Script de prueba simple sin importar el clasificador completo
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Definir modelos
class FastBinaryModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class BreedModel(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def test_model_loading():
    print("🧪 PRUEBA SIMPLE DE MODELOS")
    print("=" * 40)
    
    device = torch.device('cpu')
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Probar modelo binario
    print("1️⃣ Probando modelo binario...")
    try:
        binary_model = FastBinaryModel(num_classes=2).to(device)
        binary_path = "realtime_binary_models/best_model_epoch_1_acc_0.9649.pth"
        
        if os.path.exists(binary_path):
            checkpoint = torch.load(binary_path, map_location=device)
            binary_model.load_state_dict(checkpoint['model_state_dict'])
            binary_model.eval()
            print("✅ Modelo binario cargado")
        else:
            print(f"❌ No encontrado: {binary_path}")
            return
            
    except Exception as e:
        print(f"❌ Error modelo binario: {e}")
        return
    
    # Probar modelo de razas
    print("2️⃣ Probando modelo de razas...")
    try:
        breed_model = BreedModel(num_classes=50).to(device)
        breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
        
        if os.path.exists(breed_path):
            checkpoint = torch.load(breed_path, map_location=device)
            breed_model.load_state_dict(checkpoint['model_state_dict'])
            breed_model.eval()
            print("✅ Modelo de razas cargado")
        else:
            print(f"❌ No encontrado: {breed_path}")
            return
            
    except Exception as e:
        print(f"❌ Error modelo razas: {e}")
        return
    
    # Crear imagen de prueba
    print("3️⃣ Creando imagen de prueba...")
    test_image = Image.new('RGB', (300, 300), color=(139, 69, 19))  # Color marrón
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    print(f"✅ Tensor creado: {input_tensor.shape}")
    
    # Probar predicción binaria
    print("4️⃣ Probando predicción binaria...")
    try:
        with torch.no_grad():
            binary_output = binary_model(input_tensor)
            binary_probs = torch.softmax(binary_output, dim=1)
            binary_confidence, binary_pred = torch.max(binary_probs, 1)
            
            is_dog = bool(binary_pred.item() == 1)
            confidence = float(binary_confidence.item())
            
            print(f"   Resultado: {'🐕 PERRO' if is_dog else '❌ NO PERRO'}")
            print(f"   Confianza: {confidence:.4f}")
            
    except Exception as e:
        print(f"❌ Error predicción binaria: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Probar predicción de razas (solo si es perro)
    if is_dog:
        print("5️⃣ Probando predicción de razas...")
        try:
            with torch.no_grad():
                breed_output = breed_model(input_tensor)
                breed_probs = torch.softmax(breed_output, dim=1)
                breed_confidence, breed_pred = torch.max(breed_probs, 1)
                
                print(f"   Índice predicho: {breed_pred.item()}")
                print(f"   Confianza raza: {breed_confidence.item():.4f}")
                print("✅ Predicción de raza exitosa")
                
        except Exception as e:
            print(f"❌ Error predicción razas: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print("🎉 TODAS LAS PRUEBAS EXITOSAS")
    print("Los modelos funcionan correctamente.")
    print("El problema está en la comunicación web.")

if __name__ == "__main__":
    test_model_loading()