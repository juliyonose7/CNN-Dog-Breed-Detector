# !/usr/bin/env python3
"""
Script for inspeccionar the model best_model_fold_0.pth
"""

import torch
from pathlib import Path

def inspect_model():
    """Inspeccionar the contenido of the model saved"""
    model_path = "best_model_fold_0.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    print(f"🔍 Inspeccionando modelo: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"📋 Tipo del checkpoint: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"🔑 Claves disponibles en el checkpoint:")
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  - {key}: {type(value)} = {value}")
        else:
            print(f"⚠️ El checkpoint es directamente un estado de modelo")
            print(f"Tipo: {type(checkpoint)}")
        
        print("\n" + "="*50)
        
        # Intentar verify if es a state_dict directo
        if hasattr(checkpoint, 'keys'):
            sample_keys = list(checkpoint.keys())[:5]
            print(f"🔍 Primeras 5 claves: {sample_keys}")
        
    except Exception as e:
        print(f"❌ Error inspeccionando modelo: {e}")

if __name__ == "__main__":
    inspect_model()