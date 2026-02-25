#!/usr/bin/env python3
"""
Debug: Verificar qué razas se están cargando realmente
"""

import os
import torch

def check_breed_loading():
    print("🔍 VERIFICANDO CARGA DE RAZAS")
    print("=" * 50)
    
    # 1. Verificar directorio breed_processed_data
    breed_dir = "breed_processed_data/train"
    if os.path.exists(breed_dir):
        actual_breeds = sorted([d for d in os.listdir(breed_dir) 
                               if os.path.isdir(os.path.join(breed_dir, d))])
        print(f"📁 Razas en {breed_dir}: {len(actual_breeds)}")
        print("Primeras 10 razas:")
        for i, breed in enumerate(actual_breeds[:10]):
            print(f"   {i+1:2d}. {breed}")
        
        if len(actual_breeds) > 10:
            print(f"   ... y {len(actual_breeds)-10} más")
    else:
        print(f"❌ No existe: {breed_dir}")
    
    # 2. Verificar modelo de razas
    breed_model_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    if os.path.exists(breed_model_path):
        print(f"\n📦 Cargando modelo: {breed_model_path}")
        checkpoint = torch.load(breed_model_path, map_location='cpu')
        
        print("🔑 Claves en checkpoint:")
        for key in checkpoint.keys():
            print(f"   - {key}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Buscar la capa final
            final_layer_key = None
            for key in state_dict.keys():
                if 'fc' in key and 'weight' in key:
                    final_layer_key = key
                    break
            
            if final_layer_key:
                final_weights = state_dict[final_layer_key]
                print(f"\n🎯 Capa final encontrada: {final_layer_key}")
                print(f"   Shape: {final_weights.shape}")
                print(f"   Clases en modelo: {final_weights.shape[0]}")
            
        # Verificar si hay breed_names guardados
        if 'breed_names' in checkpoint:
            saved_breeds = checkpoint['breed_names']
            print(f"\n📋 Razas guardadas en modelo: {len(saved_breeds)}")
            print("Primeras 10:")
            for i, breed in enumerate(saved_breeds[:10]):
                print(f"   {i+1:2d}. {breed}")
        else:
            print("\n⚠️ No hay 'breed_names' en el checkpoint")
    
    # 3. Verificar dataset original
    yesdog_dir = "DATASETS/YESDOG"
    if os.path.exists(yesdog_dir):
        original_breeds = sorted([d for d in os.listdir(yesdog_dir) 
                                 if os.path.isdir(os.path.join(yesdog_dir, d))])
        print(f"\n📊 Dataset original: {len(original_breeds)} razas")
        
        # Buscar algunas razas específicas
        search_breeds = ['pug', 'labrador', 'norwegian', 'beagle']
        print(f"\n🔎 Buscando razas específicas:")
        for search in search_breeds:
            found = [b for b in original_breeds if search.lower() in b.lower()]
            print(f"   '{search}': {found}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_breed_loading()