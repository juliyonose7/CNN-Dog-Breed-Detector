"""
🐕 INICIADOR of training BINARIO
Entrena the model binario (dog vs no-dog) with control of parada manual
"""

import os
import sys
from binary_trainer import (
    optimize_for_7800x3d, 
    BinaryDogClassifier, 
    BinaryTrainer,
    create_dataloaders,
    get_transforms
)
import torch
from pathlib import Path

def main():
    """Function main"""
    print("🐕 INICIANDO ENTRENAMIENTO BINARIO CON CONTROL MANUAL")
    print("🚀 Optimizado para AMD 7800X3D")
    print("=" * 80)
    
    # Optimizar for 7800X3D
    optimize_for_7800x3d()
    
    # Configuration
    DATA_PATH = "./DATASETS"
    BATCH_SIZE = 32  # Implementation note.
    NUM_WORKERS = 12  # For 7800X3D
    
    # Verify data
    if not Path(DATA_PATH).exists():
        print(f"❌ Directorio de datos no encontrado: {DATA_PATH}")
        return
    
    # Create dataloaders
    print("📊 Cargando datasets...")
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = create_dataloaders(
        DATA_PATH, train_transform, val_transform, BATCH_SIZE, NUM_WORKERS
    )
    
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Val samples: {len(val_loader.dataset)}")
    print()
    
    # Create model
    print("🤖 Creando modelo EfficientNet-B1...")
    model = BinaryDogClassifier(pretrained=True)
    device = torch.device('cpu')  # Usando CPU for consistencia
    
    # Create trainer
    trainer = BinaryTrainer(model, device=device)
    
    print()
    print("🎯 CONFIGURACIÓN DE ENTRENAMIENTO:")
    print("   - Épocas: 25 (con early stopping)")
    print("   - Paciencia: 5 épocas sin mejora")
    print("   - Optimizador: AdamW con OneCycleLR")
    print("   - Control manual: Presiona 'q' para parar")
    print()
    
    # Entrenar model
    results = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=25,
        save_path='./binary_models',
        patience=5
    )
    
    print("🎉 ENTRENAMIENTO COMPLETADO!")
    print("=" * 80)
    print(f"✅ Mejor accuracy: {results['best_accuracy']:.2f}%")
    print(f"📅 Épocas entrenadas: {results['final_epoch']}")
    print(f"💾 Modelo guardado en: ./binary_models/best_binary_model.pth")
    print()
    print("🔄 Para copiar el modelo a la ubicación esperada:")
    print("   copy binary_models\\best_binary_model.pth best_model.pth")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()