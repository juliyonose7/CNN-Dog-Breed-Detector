"""
Script optimizado para entrenamiento rápido en CPU
Versión de prueba con dataset reducido
"""

from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
import argparse

def quick_train_cpu(dataset_path: str, epochs: int = 5):
    """Entrenamiento rápido optimizado para CPU"""
    print("⚡ ENTRENAMIENTO RÁPIDO OPTIMIZADO PARA CPU")
    print("="*50)
    
    # 1. Preprocesamiento con dataset reducido
    print("📊 Preparando dataset reducido...")
    preprocessor = DataPreprocessor(dataset_path, "./quick_processed_data")
    
    # Recolectar solo una muestra pequeña
    image_paths, labels = preprocessor.collect_all_images()
    
    # Tomar solo 1000 imágenes por clase para prueba rápida
    dog_indices = [i for i, label in enumerate(labels) if label == 1][:1000]
    nodog_indices = [i for i, label in enumerate(labels) if label == 0][:1000]
    
    selected_indices = dog_indices + nodog_indices
    quick_image_paths = [image_paths[i] for i in selected_indices]
    quick_labels = [labels[i] for i in selected_indices]
    
    print(f"✅ Usando {len(quick_image_paths)} imágenes para entrenamiento rápido")
    
    # Balancear y dividir
    balanced_paths, balanced_labels = preprocessor.balance_classes(quick_image_paths, quick_labels, 'undersample')
    splits = preprocessor.create_train_val_test_split(balanced_paths, balanced_labels)
    
    # DataLoaders optimizados para CPU
    data_loaders = preprocessor.create_data_loaders(splits, batch_size=16, num_workers=0)  # num_workers=0 para CPU
    
    print(f"📊 Dataset preparado:")
    print(f"   Train: {len(data_loaders['train'])} batches")
    print(f"   Val: {len(data_loaders['val'])} batches")
    
    # 2. Entrenamiento optimizado
    print(f"\n🤖 Iniciando entrenamiento ({epochs} épocas)...")
    
    trainer = ModelTrainer(model_name='resnet50')  # ResNet50 es más rápido que EfficientNet
    trainer.setup_training(data_loaders['train'], data_loaders['val'])
    
    # Entrenamiento con configuración CPU-optimizada
    history = trainer.train_model(
        num_epochs=epochs,
        save_path='./quick_models',
        freeze_epochs=2  # Menos épocas congeladas
    )
    
    print("\n🎉 ¡Entrenamiento rápido completado!")
    
    # Estimación para dataset completo
    train_batches_quick = len(data_loaders['train'])
    train_batches_full = 900  # Dataset completo
    scale_factor = train_batches_full / train_batches_quick
    
    print(f"\n📊 ESTIMACIÓN PARA DATASET COMPLETO:")
    print(f"   Dataset actual: {train_batches_quick} batches")
    print(f"   Dataset completo: {train_batches_full} batches")
    print(f"   Factor de escala: {scale_factor:.1f}x")
    print(f"   Tiempo estimado para dataset completo: {scale_factor * epochs / 5:.1f}x el tiempo actual")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento rápido para pruebas")
    parser.add_argument("--dataset", required=True, help="Ruta al directorio DATASETS")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas")
    
    args = parser.parse_args()
    
    quick_train_cpu(args.dataset, args.epochs)