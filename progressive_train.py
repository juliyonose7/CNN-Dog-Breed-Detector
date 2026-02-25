"""
Script of improvement progresiva of the model
Estrategias for optimizar performance paso a paso
"""

import argparse
from quick_train import quick_train_cpu
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer

def progressive_improvement(dataset_path: str, stage: int = 1):
    """Improvement progresiva of the model en etapas"""
    
    stages = {
        1: {
            "name": "🟢 Básico - Más épocas",
            "samples_per_class": 1000,
            "epochs": 10,
            "batch_size": 16,
            "model": "resnet50"
        },
        2: {
            "name": "🟡 Intermedio - Más datos",
            "samples_per_class": 3000,
            "epochs": 8,
            "batch_size": 16,
            "model": "resnet50"
        },
        3: {
            "name": "🟠 Avanzado - Mejor modelo",
            "samples_per_class": 5000,
            "epochs": 10,
            "batch_size": 12,
            "model": "efficientnet_b3"
        },
        4: {
            "name": "🔴 Máximo - Dataset completo",
            "samples_per_class": None,  # Todo the dataset
            "epochs": 20,
            "batch_size": 8,
            "model": "efficientnet_b3"
        }
    }
    
    config = stages[stage]
    print(f"🚀 {config['name']}")
    print("="*60)
    
    # Implementation note.
    preprocessor = DataPreprocessor(dataset_path, f"./stage_{stage}_processed")
    image_paths, labels = preprocessor.collect_all_images()
    
    if config["samples_per_class"]:
        # Use shows limitada
        dog_indices = [i for i, label in enumerate(labels) if label == 1][:config["samples_per_class"]]
        nodog_indices = [i for i, label in enumerate(labels) if label == 0][:config["samples_per_class"]]
        
        selected_indices = dog_indices + nodog_indices
        image_paths = [image_paths[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
        
        print(f"📊 Usando {len(image_paths)} imágenes ({config['samples_per_class']} por clase)")
    else:
        print(f"📊 Usando dataset completo: {len(image_paths)} imágenes")
    
    # Balancear and dividir
    balanced_paths, balanced_labels = preprocessor.balance_classes(image_paths, labels, 'undersample')
    splits = preprocessor.create_train_val_test_split(balanced_paths, balanced_labels)
    
    # DataLoaders
    data_loaders = preprocessor.create_data_loaders(
        splits, 
        batch_size=config["batch_size"], 
        num_workers=0
    )
    
    # Training
    trainer = ModelTrainer(model_name=config["model"])
    trainer.setup_training(data_loaders['train'], data_loaders['val'])
    
    history = trainer.train_model(
        num_epochs=config["epochs"],
        save_path=f'./stage_{stage}_models',
        freeze_epochs=3
    )
    
    # Show improvement
    best_acc = max(history['val_accuracy'])
    print(f"\n🎯 RESULTADO ETAPA {stage}:")
    print(f"   Mejor accuracy: {best_acc:.4f}")
    print(f"   Modelo: {config['model']}")
    print(f"   Épocas: {config['epochs']}")
    
    # Implementation note.
    if stage < 4:
        print(f"\n💡 SIGUIENTE PASO:")
        print(f"   python progressive_train.py --dataset \".\\DATASETS\" --stage {stage + 1}")
        
        # Implementation note.
        if stage == 1:
            print(f"   Tiempo estimado: 15-20 minutos")
        elif stage == 2:
            print(f"   Tiempo estimado: 45-60 minutos")
        elif stage == 3:
            print(f"   Tiempo estimado: 2-3 horas")
    else:
        print(f"\n🏆 ¡ENTRENAMIENTO COMPLETO FINALIZADO!")
        print(f"   Tu modelo está listo para producción")

def compare_models():
    """Compara performance of diferentes etapas"""
    import json
    import os
    
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*40)
    
    for stage in range(1, 5):
        history_path = f"./stage_{stage}_models/training_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            best_acc = max(history['val_accuracy'])
            print(f"Etapa {stage}: {best_acc:.4f} accuracy")
        else:
            print(f"Etapa {stage}: No entrenada")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mejora progresiva del modelo")
    parser.add_argument("--dataset", required=True, help="Ruta al directorio DATASETS")
    parser.add_argument("--stage", type=int, default=1, choices=[1,2,3,4], 
                       help="Etapa de mejora (1=básico, 4=completo)")
    parser.add_argument("--compare", action="store_true", help="Comparar modelos existentes")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        progressive_improvement(args.dataset, args.stage)