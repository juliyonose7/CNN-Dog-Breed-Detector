#!/usr/bin/env python3
"""
Model Checkpoint Inspector
==========================

Utility script for inspecting PyTorch model checkpoint files (.pth).
Displays checkpoint structure, tensor shapes, and metadata to help
understand model architecture and training state.

Usage:
    python inspect_model_checkpoint.py

Author: Dog Classification Team
Version: 1.0.0
"""

import torch
from pathlib import Path


def inspect_model():
    """
    Inspect the contents of a saved model checkpoint.
    
    Loads the checkpoint file and displays:
        - Checkpoint type (dict or state_dict)
        - Available keys in the checkpoint
        - Tensor shapes and data types
        - Other metadata values
    
    Returns:
        None: Prints inspection results to stdout.
    """
    model_path = "best_model_fold_0.pth"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    
    print(f"Inspecting model: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Available keys in checkpoint:")
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  - {key}: {type(value)} = {value}")
        else:
            print(f"Warning: Checkpoint is directly a model state")
            print(f"Type: {type(checkpoint)}")
        
        print("\n" + "="*50)
        
        # Try to verify if it's a direct state_dict
        if hasattr(checkpoint, 'keys'):
            sample_keys = list(checkpoint.keys())[:5]
            print(f"First 5 keys: {sample_keys}")
        
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_model()