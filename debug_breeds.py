# !/usr/bin/env python3
"""
Breed Model Loading Verification Module.

This module provides diagnostic tools for verifying the correct loading and
configuration of the breed classification model. It validates the consistency
between the model's output layer, saved breed names, and the actual dataset
structure.

Key Verifications:
    - Processed dataset directory structure
    - Model checkpoint loading and structure
    - Breed name mappings consistency
    - Original dataset breed availability

Usage:
    python debug_breeds.py

Author: System IA
Date: 2024
"""

import os
import torch


def check_breed_loading():
    """
    Perform comprehensive breed loading verification.

    This function verifies the consistency between:
    1. The processed training data directory structure
    2. The saved model checkpoint and its breed mappings
    3. The original dataset structure

    It helps identify mismatches between model outputs and expected breeds,
    which can cause prediction errors.

    Returns:
        None: Prints diagnostic information to stdout.

    Checks Performed:
        - Counts breeds in processed data directory
        - Loads and inspects model checkpoint keys
        - Identifies final classification layer dimensions
        - Compares saved breed names with directory structure
    """
    print("🔍 VERIFYING BREED LOADING")
    print("=" * 50)
    
    # 1. Verify processed breed data directory
    breed_dir = "breed_processed_data/train"
    if os.path.exists(breed_dir):
        actual_breeds = sorted([d for d in os.listdir(breed_dir) 
                               if os.path.isdir(os.path.join(breed_dir, d))])
        print(f"📁 Breeds in {breed_dir}: {len(actual_breeds)}")
        print("First 10 breeds:")
        for i, breed in enumerate(actual_breeds[:10]):
            print(f"   {i+1:2d}. {breed}")
        
        if len(actual_breeds) > 10:
            print(f"   ... and {len(actual_breeds)-10} more")
    else:
        print(f"❌ Directory not found: {breed_dir}")
    
    # 2. Verify breed classification model
    breed_model_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    if os.path.exists(breed_model_path):
        print(f"\n📦 Loading model: {breed_model_path}")
        checkpoint = torch.load(breed_model_path, map_location='cpu')
        
        print("🔑 Keys in checkpoint:")
        for key in checkpoint.keys():
            print(f"   - {key}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Search for the final classification layer
            final_layer_key = None
            for key in state_dict.keys():
                if 'fc' in key and 'weight' in key:
                    final_layer_key = key
                    break
            
            if final_layer_key:
                final_weights = state_dict[final_layer_key]
                print(f"\n🎯 Final layer found: {final_layer_key}")
                print(f"   Shape: {final_weights.shape}")
                print(f"   Number of classes in model: {final_weights.shape[0]}")
            
        # Verify if breed_names are saved in checkpoint
        if 'breed_names' in checkpoint:
            saved_breeds = checkpoint['breed_names']
            print(f"\n📋 Breeds saved in model: {len(saved_breeds)}")
            print("First 10:")
            for i, breed in enumerate(saved_breeds[:10]):
                print(f"   {i+1:2d}. {breed}")
        else:
            print("\n⚠️ No 'breed_names' in checkpoint")
    
    # 3. Verify original dataset
    yesdog_dir = "DATASETS/YESDOG"
    if os.path.exists(yesdog_dir):
        original_breeds = sorted([d for d in os.listdir(yesdog_dir) 
                                 if os.path.isdir(os.path.join(yesdog_dir, d))])
        print(f"\n📊 Original dataset: {len(original_breeds)} breeds")
        
        # Search for specific breeds
        search_breeds = ['pug', 'labrador', 'norwegian', 'beagle']
        print(f"\n🔎 Searching for specific breeds:")
        for search in search_breeds:
            found = [b for b in original_breeds if search.lower() in b.lower()]
            print(f"   '{search}': {found}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    check_breed_loading()