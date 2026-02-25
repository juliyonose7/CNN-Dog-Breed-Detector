#!/usr/bin/env python3
"""
Script de diagnóstico para verificar los modelos guardados
"""

import torch
import os

def check_model(model_path, model_name):
    print(f"\n🔍 Verificando {model_name}: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Archivo no existe: {model_path}")
        return
        
    try:
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"✅ Archivo cargado exitosamente")
        print(f"📋 Keys disponibles: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Buscar la capa final para ver cuántas clases
            for key, param in state_dict.items():
                if 'fc' in key and 'weight' in key:
                    print(f"🎯 Capa final ({key}): {param.shape}")
                    print(f"📊 Número de clases detectado: {param.shape[0]}")
        
        # Verificar métricas si están disponibles
        metrics = ['val_accuracy', 'accuracy', 'best_acc']
        for metric in metrics:
            if metric in checkpoint:
                print(f"📈 {metric}: {checkpoint[metric]}")
                
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

# Verificar ambos modelos
print("🔍 DIAGNÓSTICO DE MODELOS")
print("=" * 50)

# Modelo binario
check_model("realtime_binary_models/best_model_epoch_1_acc_0.9649.pth", "Modelo Binario")

# Modelo de razas
check_model("autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth", "Modelo de Razas")

print("\n" + "=" * 50)
print("✅ Diagnóstico completado")