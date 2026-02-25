"""
Binary Training Launcher Script.

Launches binary dog classifier training (dog vs no-dog) with manual stop control.
Optimized for AMD 7800X3D CPU with EfficientNet-B1 architecture.

Features:
    - Manual stop control during training
    - Early stopping with configurable patience
    - AdamW optimizer with OneCycleLR scheduler
    - Optimized worker count for 7800X3D
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
    """Main entry point for binary training."""
    print(" LAUNCHING BINARY TRAINING WITH MANUAL CONTROL")
    print(" Optimized for AMD 7800X3D")
    print("=" * 80)
    
    # Optimize for 7800X3D
    optimize_for_7800x3d()
    
    # Configuration
    DATA_PATH = "./DATASETS"
    BATCH_SIZE = 32  # Larger batch for efficiency
    NUM_WORKERS = 12  # For 7800X3D
    
    # Verify data
    if not Path(DATA_PATH).exists():
        print(f" Data directory not found: {DATA_PATH}")
        return
    
    # Create dataloaders
    print(" Loading datasets...")
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = create_dataloaders(
        DATA_PATH, train_transform, val_transform, BATCH_SIZE, NUM_WORKERS
    )
    
    print(f" Train samples: {len(train_loader.dataset)}")
    print(f" Val samples: {len(val_loader.dataset)}")
    print()
    
    # Create model
    print(" Creating EfficientNet-B1 model...")
    model = BinaryDogClassifier(pretrained=True)
    device = torch.device('cpu')  # Using CPU for consistency
    
    # Create trainer
    trainer = BinaryTrainer(model, device=device)
    
    print()
    print(" TRAINING CONFIGURATION:")
    print("   - Epochs: 25 (with early stopping)")
    print("   - Patience: 5 epochs without improvement")
    print("   - Optimizer: AdamW with OneCycleLR")
    print("   - Manual control: Press 'q' to stop")
    print()
    
    # Train model
    results = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=25,
        save_path='./binary_models',
        patience=5
    )
    
    print(" TRAINING COMPLETED!")
    print("=" * 80)
    print(f" Best accuracy: {results['best_accuracy']:.2f}%")
    print(f" Epochs trained: {results['final_epoch']}")
    print(f" Model saved at: ./binary_models/best_binary_model.pth")
    print()
    print(" To copy the model to the expected location:")
    print("   copy binary_models\\best_binary_model.pth best_model.pth"))
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()