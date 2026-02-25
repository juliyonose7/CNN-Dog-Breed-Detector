"""
Quick Training Script Optimized for CPU.

This module provides a fast training pipeline optimized for CPU execution.
It uses a reduced dataset subset for quick experimentation and model
validation without requiring GPU resources.

Features:
    - Reduced dataset sampling (1000 images per class by default)
    - CPU-optimized DataLoader configuration (num_workers=0)
    - ResNet50 backbone for efficient feature extraction
    - Training time estimation for full dataset

Usage:
    python quick_train.py --dataset "./DATASETS" --epochs 5

Author: AI System
Date: 2024
"""

from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
import argparse


def quick_train_cpu(dataset_path: str, epochs: int = 5):
    """
    Execute quick training optimized for CPU execution.
    
    Performs a fast training cycle using a subset of the dataset,
    optimized for CPU execution without GPU acceleration. Useful
    for quick testing and validation.
    
    Args:
        dataset_path: Path to the DATASETS directory containing images.
        epochs: Number of training epochs. Defaults to 5.
    
    Returns:
        None. Prints training results and time estimates.
    """
    print("⚡ QUICK TRAINING OPTIMIZED FOR CPU")
    print("="*50)
    
    # 1. Preprocessing with reduced dataset
    print("📊 Preparing reduced dataset...")
    preprocessor = DataPreprocessor(dataset_path, "./quick_processed_data")
    
    # Collect all image paths
    image_paths, labels = preprocessor.collect_all_images()
    
    # Limit to 1000 samples per class for quick training
    dog_indices = [i for i, label in enumerate(labels) if label == 1][:1000]
    nodog_indices = [i for i, label in enumerate(labels) if label == 0][:1000]
    
    selected_indices = dog_indices + nodog_indices
    quick_image_paths = [image_paths[i] for i in selected_indices]
    quick_labels = [labels[i] for i in selected_indices]
    
    print(f"✅ Using {len(quick_image_paths)} images for quick training")
    
    # Balance and split dataset
    balanced_paths, balanced_labels = preprocessor.balance_classes(quick_image_paths, quick_labels, 'undersample')
    splits = preprocessor.create_train_val_test_split(balanced_paths, balanced_labels)
    
    # Create DataLoaders optimized for CPU (num_workers=0)
    data_loaders = preprocessor.create_data_loaders(splits, batch_size=16, num_workers=0)
    
    print(f"📊 Dataset prepared:")
    print(f"   Train: {len(data_loaders['train'])} batches")
    print(f"   Val: {len(data_loaders['val'])} batches")
    
    # 2. Optimized training
    print(f"\n🤖 Starting training ({epochs} epochs)...")
    
    trainer = ModelTrainer(model_name='resnet50')  # More efficient for quick training
    trainer.setup_training(data_loaders['train'], data_loaders['val'])
    
    # Train with CPU-optimized configuration
    history = trainer.train_model(
        num_epochs=epochs,
        save_path='./quick_models',
        freeze_epochs=2  # Freeze fewer epochs for fast experimentation
    )
    
    print("\n🎉 Quick training completed!")
    
    # Estimate time for full dataset
    train_batches_quick = len(data_loaders['train'])
    train_batches_full = 900  # Full dataset batch count
    scale_factor = train_batches_full / train_batches_quick
    
    print(f"\n📊 ESTIMATION FOR FULL DATASET:")
    print(f"   Current dataset: {train_batches_quick} batches")
    print(f"   Full dataset: {train_batches_full} batches")
    print(f"   Scale factor: {scale_factor:.1f}x")
    print(f"   Estimated time for full dataset: {scale_factor * epochs / 5:.1f}x current time")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick training for testing")
    parser.add_argument("--dataset", required=True, help="Path to DATASETS directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    
    args = parser.parse_args()
    
    quick_train_cpu(args.dataset, args.epochs)