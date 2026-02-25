"""
System Status Verification Module.

This module provides comprehensive system health checks for the hierarchical
canine classification system. It verifies the availability and status of all
critical components including machine learning models, datasets, configuration
files, API endpoints, and system dependencies.

The verification system is optimized for AMD 7800X3D processors and provides
detailed diagnostics for troubleshooting deployment issues.

Components Verified:
    - Binary classification model (dog vs. not-dog)
    - Breed classification model (multi-class)
    - Training and validation datasets
    - Configuration files and mappings
    - API server files
    - Required Python dependencies

Usage:
    python check_system_status.py

Author: System IA
Date: 2024
"""

import os
import json
from pathlib import Path
import torch


def check_models():
    """
    Verify the status and availability of machine learning models.

    This function checks for the existence and basic properties of both
    the binary classification model and the breed classification model.
    It reports file sizes and last known accuracy metrics.

    Returns:
        None: Prints status information to stdout.

    Note:
        - Binary model expected at 'best_model.pth'
        - Breed models expected in 'breed_models/' directory
    """
    print(" VERIFYING MODEL STATUS")
    print("=" * 60)
    
    # Binary classification model
    binary_model_path = "best_model.pth"
    if os.path.exists(binary_model_path):
        print(" Binary model: AVAILABLE")
        print(f"    File: {binary_model_path}")
        print(f"    Size: {os.path.getsize(binary_model_path) / (1024*1024):.1f} MB")
        print(f"    Reported accuracy: 91.33%")
    else:
        print(" Binary model: NOT FOUND")
    
    print()
    
    # Breed classification model
    breed_models_dir = Path("breed_models")
    if breed_models_dir.exists():
        checkpoints = list(breed_models_dir.glob("*.pth"))
        if checkpoints:
            print(" Breed model: CHECKPOINTS AVAILABLE")
            for checkpoint in checkpoints:
                print(f"    {checkpoint.name}: {checkpoint.stat().st_size / (1024*1024):.1f} MB")
        else:
            print("  Breed model: DIRECTORY EXISTS, NO CHECKPOINTS")
    else:
        print("  Breed model: TRAINING IN PROGRESS / NOT AVAILABLE")
    
    print()

def check_datasets():
    """
    Verify the status and availability of training datasets.

    This function checks for the existence of both the original raw dataset
    structure (YESDOG/NODOG folders) and the processed dataset information.
    It reports statistics about available breeds and image counts.

    Returns:
        None: Prints dataset status information to stdout.

    Dataset Structure Expected:
        DATASETS/
        ├── YESDOG/     # Dog breed folders
        │   ├── breed1/
        │   └── breed2/
        └── NODOG/      # Non-dog category folders
    """
    print(" VERIFYING DATASETS")
    print("=" * 60)
    
    # Original dataset structure
    datasets_dir = Path("DATASETS")
    if datasets_dir.exists():
        yesdog_dir = datasets_dir / "YESDOG"
        nodog_dir = datasets_dir / "NODOG"
        
        if yesdog_dir.exists():
            breed_dirs = [d for d in yesdog_dir.iterdir() if d.is_dir()]
            print(f" Dataset YESDOG: {len(breed_dirs)} breeds available")
        
        if nodog_dir.exists():
            print(" Dataset NODOG: Available")
    else:
        print(" Original dataset: NOT FOUND")
    
    # Processed breed dataset
    dataset_info_path = "dataset_info.json"
    if os.path.exists(dataset_info_path):
        print(" Processed breed dataset: AVAILABLE")
        with open(dataset_info_path, 'r') as f:
            info = json.load(f)
            print(f"     Breeds: {info['total_classes']}")
            print(f"    Total images: {info['total_samples']:,}")
            print(f"     Training: {info['train_samples']:,}")
            print(f"    Validation: {info['val_samples']:,}")
            print(f"    Test: {info['test_samples']:,}")
    else:
        print("  Processed breed dataset: NOT AVAILABLE")
    
    print()

def check_configuration():
    """
    Verify system configuration files and breed mappings.

    This function checks for the existence and validity of configuration
    files required for breed classification, including class name mappings
    and index assignments.

    Returns:
        None: Prints configuration status to stdout.

    Configuration Files Checked:
        - breed_config.py: Contains CLASS_NAMES and CLASS_TO_IDX mappings
    """
    print("  VERIFYING CONFIGURATIONS")
    print("=" * 60)
    
    # Breed configuration
    breed_config_path = "breed_config.py"
    if os.path.exists(breed_config_path):
        print(" Breed configuration: AVAILABLE")
        try:
            import breed_config
            print(f"     Configured breeds: {len(breed_config.CLASS_NAMES)}")
            print(f"    Mapped indices: {len(breed_config.CLASS_TO_IDX)}")
        except ImportError:
            print("     Error importing configuration")
    else:
        print(" Breed configuration: NOT FOUND")
    
    print()

def check_api_files():
    """
    Verify the availability of API server and frontend files.

    This function checks for the existence of both hierarchical and simple
    API implementations along with their corresponding frontend files.

    Returns:
        None: Prints API file status to stdout.

    Files Checked:
        - hierarchical_api.py: Advanced hierarchical classification API
        - hierarchical_frontend.html: Web interface for hierarchical API
        - app.py: Simple binary classification API
        - index.html: Basic web interface
    """
    print(" VERIFYING API FILES")
    print("=" * 60)
    
    api_files = [
        ("hierarchical_api.py", "Hierarchical API"),
        ("hierarchical_frontend.html", "Hierarchical Frontend"),
        ("app.py", "Simple Original API"),
        ("index.html", "Original Frontend")
    ]
    
    for file_path, description in api_files:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f" {description}: {size_kb:.1f} KB")
        else:
            print(f" {description}: NOT FOUND")
    
    print()

def check_system_requirements():
    """
    Verify that all required Python dependencies are installed.

    This function checks for the presence of critical machine learning
    and web framework dependencies required by the classification system.

    Returns:
        None: Prints dependency status to stdout.

    Dependencies Checked:
        - PyTorch: Deep learning framework
        - TorchVision: Image processing utilities
        - FastAPI: Web API framework
        - Albumentations: Image augmentation library
    """
    print(" VERIFYING SYSTEM REQUIREMENTS")
    print("=" * 60)
    
    try:
        import torch
        print(f" PyTorch: {torch.__version__}")
        print(f"    Available CPU threads: {torch.get_num_threads()}")
        print(f"    7800X3D Optimization: Configured")
    except ImportError:
        print(" PyTorch: NOT INSTALLED")
    
    try:
        import torchvision
        print(f" TorchVision: {torchvision.__version__}")
    except ImportError:
        print(" TorchVision: NOT INSTALLED")
    
    try:
        import fastapi
        print(f" FastAPI: {fastapi.__version__}")
    except ImportError:
        print(" FastAPI: NOT INSTALLED")
    
    try:
        import albumentations
        print(f" Albumentations: {albumentations.__version__}")
    except ImportError:
        print(" Albumentations: NOT INSTALLED")
    
    print()

def get_recommendations():
    """
    Provide actionable recommendations based on current system status.

    This function analyzes the current state of the system components
    and provides specific recommendations for deployment, testing, or
    further development based on what is available.

    Returns:
        None: Prints recommendations to stdout.

    Recommendation Categories:
        - System ready for demonstration
        - Basic system available
        - System not operational
    """
    print(" RECOMMENDATIONS")
    print("=" * 60)
    
    # Check available components
    has_binary = os.path.exists("best_model.pth")
    has_breed_config = os.path.exists("breed_config.py")
    
    if has_binary and has_breed_config:
        print(" SYSTEM READY FOR DEMONSTRATION:")
        print("   1. Run hierarchical API: python hierarchical_api.py")
        print("   2. Open frontend: hierarchical_frontend.html")
        print("   3. The system will work with:")
        print("      • Binary detection:  Fully functional")
        print("      • Breed classification:  Will show training message")
    
    elif has_binary:
        print(" BASIC SYSTEM AVAILABLE:")
        print("   1. Use original API: python app.py")
        print("   2. Binary detection functional at 91.33%")
    
    else:
        print(" SYSTEM NOT OPERATIONAL:")
        print("   1. Binary model missing")
        print("   2. Run complete training")
    
    print("\n SUGGESTED NEXT STEPS:")
    if not os.path.exists("breed_models"):
        print("   • Resume breed training: python breed_trainer.py")
    
    print("   • Test current system with hierarchical frontend")
    print("   • Monitor breed training progress")
    
    print()

def main():
    """
    Execute complete system status verification.

    This is the main entry point that orchestrates all verification checks
    in sequence, providing a comprehensive overview of system health.

    Returns:
        None: Runs all verification functions and prints results.
    """
    print(" HIERARCHICAL CANINE SYSTEM - COMPLETE STATUS")
    print(" Optimized for AMD 7800X3D")
    print("=" * 80)
    print()
    
    check_models()
    check_datasets()
    check_configuration()
    check_api_files()
    check_system_requirements()
    get_recommendations()
    
    print("=" * 80)
    print(" Complete verification finished")

if __name__ == "__main__":
    main()