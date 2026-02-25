"""
Progressive Model Improvement Training Script.

This module implements a staged progressive training approach for the dog
classification model. It allows incremental training with configurable
stages that progressively increase dataset size, model complexity, and
training duration.

Stages:
    Stage 1 (Basic): Limited samples, fewer epochs - for quick testing
    Stage 2 (Intermediate): More data with moderate training time
    Stage 3 (Advanced): Better model architecture with larger dataset
    Stage 4 (Maximum): Full dataset with complete training

Usage:
    python progressive_train.py --dataset "./DATASETS" --stage 1
    python progressive_train.py --dataset "./DATASETS" --compare

Author: AI System
Date: 2024
"""

import argparse
from quick_train import quick_train_cpu
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer


def progressive_improvement(dataset_path: str, stage: int = 1):
    """
    Execute progressive model improvement training at specified stage.
    
    Implements a staged training approach where each stage increases
    the complexity and data volume to progressively improve model
    performance.
    
    Args:
        dataset_path: Path to the DATASETS directory containing images.
        stage: Training stage (1-4), where higher stages use more data
               and longer training. Defaults to 1.
    
    Returns:
        None. Prints training results and saves model checkpoints.
    
    Raises:
        ValueError: If stage is not between 1 and 4.
    """
    stages = {
        1: {
            "name": " Basic - More epochs",
            "samples_per_class": 1000,
            "epochs": 10,
            "batch_size": 16,
            "model": "resnet50"
        },
        2: {
            "name": " Intermediate - More data",
            "samples_per_class": 3000,
            "epochs": 8,
            "batch_size": 16,
            "model": "resnet50"
        },
        3: {
            "name": " Advanced - Better model",
            "samples_per_class": 5000,
            "epochs": 10,
            "batch_size": 12,
            "model": "efficientnet_b3"
        },
        4: {
            "name": " Maximum - Full dataset",
            "samples_per_class": None,  # Use entire dataset
            "epochs": 20,
            "batch_size": 8,
            "model": "efficientnet_b3"
        }
    }
    
    config = stages[stage]
    print(f" {config['name']}")
    print("="*60)
    
    # Initialize preprocessor for current stage
    preprocessor = DataPreprocessor(dataset_path, f"./stage_{stage}_processed")
    image_paths, labels = preprocessor.collect_all_images()
    
    if config["samples_per_class"]:
        # Use limited sample size per class
        dog_indices = [i for i, label in enumerate(labels) if label == 1][:config["samples_per_class"]]
        nodog_indices = [i for i, label in enumerate(labels) if label == 0][:config["samples_per_class"]]
        
        selected_indices = dog_indices + nodog_indices
        image_paths = [image_paths[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
        
        print(f" Using {len(image_paths)} images ({config['samples_per_class']} per class)")
    else:
        print(f" Using complete dataset: {len(image_paths)} images")
    
    # Balance and split dataset
    balanced_paths, balanced_labels = preprocessor.balance_classes(image_paths, labels, 'undersample')
    splits = preprocessor.create_train_val_test_split(balanced_paths, balanced_labels)
    
    # Create DataLoaders
    data_loaders = preprocessor.create_data_loaders(
        splits, 
        batch_size=config["batch_size"], 
        num_workers=0
    )
    
    # Initialize and run training
    trainer = ModelTrainer(model_name=config["model"])
    trainer.setup_training(data_loaders['train'], data_loaders['val'])
    
    history = trainer.train_model(
        num_epochs=config["epochs"],
        save_path=f'./stage_{stage}_models',
        freeze_epochs=3
    )
    
    # Display improvement results
    best_acc = max(history['val_accuracy'])
    print(f"\n STAGE {stage} RESULTS:")
    print(f"   Best accuracy: {best_acc:.4f}")
    print(f"   Model: {config['model']}")
    print(f"   Epochs: {config['epochs']}")
    
    # Suggest next stage if not at maximum
    if stage < 4:
        print(f"\n NEXT STEP:")
        print(f"   python progressive_train.py --dataset \".\\DATASETS\" --stage {stage + 1}")
        
        # Provide estimated training time for next stage
        if stage == 1:
            print(f"   Estimated time: 15-20 minutes")
        elif stage == 2:
            print(f"   Estimated time: 45-60 minutes")
        elif stage == 3:
            print(f"   Estimated time: 2-3 hours")
    else:
        print(f"\n COMPLETE TRAINING FINISHED!")
        print(f"   Your model is ready for production")

def compare_models():
    """
    Compare performance across different training stages.
    
    Loads training history from each stage and displays the best
    validation accuracy achieved, allowing comparison of model
    improvement across progressive stages.
    
    Returns:
        None. Prints comparison results to stdout.
    """
    import json
    import os
    
    print(" MODEL COMPARISON")
    print("="*40)
    
    for stage in range(1, 5):
        history_path = f"./stage_{stage}_models/training_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            best_acc = max(history['val_accuracy'])
            print(f"Stage {stage}: {best_acc:.4f} accuracy")
        else:
            print(f"Stage {stage}: Not trained")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive model improvement training")
    parser.add_argument("--dataset", required=True, help="Path to DATASETS directory")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], 
                       help="Improvement stage (1=basic, 4=complete)")
    parser.add_argument("--compare", action="store_true", help="Compare existing models")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        progressive_improvement(args.dataset, args.stage)