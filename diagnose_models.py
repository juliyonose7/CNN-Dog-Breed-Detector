# !/usr/bin/env python3
"""
Model Diagnosis Module.

This module provides diagnostic tools for inspecting and validating
PyTorch model checkpoints. It verifies model structure, identifies
the number of output classes, and reports saved accuracy metrics
for both binary and breed classification models.

Key Features:
    - Checkpoint file loading and inspection
    - Model state dictionary analysis
    - Final layer dimension extraction
    - Accuracy metric retrieval

Usage:
    python diagnose_models.py

Author: System IA
Date: 2024
"""

import torch
import os


def check_model(model_path, model_name):
    """
    Inspect and diagnose a PyTorch model checkpoint.

    This function loads a model checkpoint and extracts diagnostic
    information including available keys, final layer dimensions,
    and reported accuracy metrics.

    Args:
        model_path (str): Path to the .pth checkpoint file.
        model_name (str): Descriptive name for the model (for display).

    Returns:
        None: Prints diagnostic information to stdout.

    Information Extracted:
        - Checkpoint file existence verification
        - Available keys in the checkpoint dictionary
        - Final fully-connected layer dimensions
        - Detected number of output classes
        - Saved accuracy metrics (if available)

    Example:
        >>> check_model("models/best_model.pth", "Binary Classifier")
    """
    print(f"\n Verifying {model_name}: {model_path}")
    
    if not os.path.exists(model_path):
        print(f" File does not exist: {model_path}")
        return
        
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f" File loaded successfully")
        print(f" Available keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Find and analyze final classification layer
            for key, param in state_dict.items():
                if 'fc' in key and 'weight' in key:
                    print(f" Final layer ({key}): {param.shape}")
                    print(f" Detected number of classes: {param.shape[0]}")
        
        # Check for saved accuracy metrics
        metrics = ['val_accuracy', 'accuracy', 'best_acc']
        for metric in metrics:
            if metric in checkpoint:
                print(f" {metric}: {checkpoint[metric]}")
                
    except Exception as e:
        print(f" Error loading model: {e}")


# Verify both models
print(" MODEL DIAGNOSTICS")
print("=" * 50)

# Binary classification model
check_model("realtime_binary_models/best_model_epoch_1_acc_0.9649.pth", "Binary Model")

# Breed classification model
check_model("autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth", "Breed Model")

print("\n" + "=" * 50)
print(" Diagnostics completed")