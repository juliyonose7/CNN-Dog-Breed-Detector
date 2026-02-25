"""
Main Training Pipeline
======================

Complete end-to-end pipeline for training a dog vs non-dog classification model.
This script orchestrates all stages of the machine learning workflow from data
analysis through model optimization and API deployment.

Pipeline Stages:
    1. Data Analysis - Analyze dataset distribution and quality
    2. Data Preprocessing - Balance, augment, and prepare data loaders
    3. Model Training - Train deep learning model with early stopping
    4. Model Optimization - Convert to TorchScript/ONNX for production
    5. API Configuration - Set up FastAPI server for inference

Usage:
    python main_pipeline.py --dataset /path/to/DATASETS --epochs 30
    python main_pipeline.py --dataset /path/to/DATASETS --skip-training --api-only

Command Line Arguments:
    --dataset: Path to the DATASETS directory (required)
    --model: Model architecture (default: efficientnet_b3)
    --epochs: Number of training epochs (default: 30)
    --balance: Balancing strategy (undersample, oversample, none)
    --skip-install: Skip dependency installation
    --skip-analysis: Skip data analysis step
    --skip-training: Skip training, use existing model
    --api-only: Only configure the API server

Author: Dog Classification Team  
Version: 1.0.0
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import time
import json

def print_header(title: str):
    """
    Print a formatted section header.
    
    Args:
        title (str): The header title text.
    """
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)


def print_step(step: str, description: str):
    """
    Print a formatted pipeline step indicator.
    
    Args:
        step (str): Step number or identifier.
        description (str): Description of the step.
    """
    print(f"\nSTEP: {step}")
    print(f"   {description}")

def run_data_analysis(dataset_path: str):
    """
    Execute dataset analysis to understand data distribution.
    
    Analyzes the dataset for class balance, image quality, and other
    statistics useful for training configuration.
    
    Args:
        dataset_path (str): Path to the dataset directory.
    
    Returns:
        bool: True if analysis completed successfully, False otherwise.
    """
    print_step("1", "Dataset Analysis")
    
    try:
        from data_analyzer import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer(dataset_path)
        analyzer.run_complete_analysis()
        
        print("Data analysis completed")
        return True
        
    except Exception as e:
        print(f"Error in data analysis: {e}")
        return False

def run_data_preprocessing(dataset_path: str, output_path: str, balance_strategy: str = 'undersample'):
    """
    Execute data preprocessing and preparation.
    
    Creates balanced datasets, applies augmentation, and prepares
    PyTorch DataLoaders for training and validation.
    
    Args:
        dataset_path (str): Path to the raw dataset directory.
        output_path (str): Path to save processed data.
        balance_strategy (str, optional): Balancing method ('undersample', 
                                         'oversample', 'none'). Defaults to 'undersample'.
    
    Returns:
        tuple: (data_loaders, splits) containing DataLoader dict and split information,
               or (None, None) if preprocessing fails.
    """
    print_step("2", "Data Preprocessing")
    
    try:
        from data_preprocessor import DataPreprocessor, create_sample_visualization
        
        preprocessor = DataPreprocessor(dataset_path, output_path)
        data_loaders, splits = preprocessor.process_complete_dataset(
            balance_strategy=balance_strategy,
            batch_size=32
        )
        
        # Create sample visualization
        sample_viz_path = Path(output_path) / 'sample_visualization.png'
        create_sample_visualization(data_loaders, str(sample_viz_path))
        
        print("Data preprocessing completed")
        return data_loaders, splits
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None

def run_model_training(data_loaders, model_name: str = 'efficientnet_b3', num_epochs: int = 30):
    """
    Execute model training.
    
    Trains the deep learning model using the specified architecture
    with transfer learning and gradual unfreezing.
    
    Args:
        data_loaders (dict): Dictionary with 'train' and 'val' DataLoaders.
        model_name (str, optional): Model architecture name. Defaults to 'efficientnet_b3'.
        num_epochs (int, optional): Number of training epochs. Defaults to 30.
    
    Returns:
        tuple: (trainer, model_path) containing ModelTrainer instance and path to best model,
               or (None, None) if training fails.
    """
    print_step("3", f"Model Training ({model_name})")
    
    try:
        from model_trainer import ModelTrainer, setup_rocm_optimization
        
        # Configure ROCm optimizations
        rocm_available = setup_rocm_optimization()
        
        # Create trainer
        trainer = ModelTrainer(model_name=model_name)
        
        # Configure training
        trainer.setup_training(data_loaders['train'], data_loaders['val'])
        
        # Train model
        models_dir = Path('./models')
        models_dir.mkdir(exist_ok=True)
        
        history = trainer.train_model(
            num_epochs=num_epochs,
            save_path=str(models_dir),
            freeze_epochs=5
        )
        
        print("Training completed")
        return trainer, str(models_dir / 'best_model.pth')
        
    except Exception as e:
        print(f"Error in training: {e}")
        return None, None

def run_model_optimization(model_path: str):
    """
    Execute model optimization for production deployment.
    
    Converts the trained model to TorchScript and ONNX formats,
    runs benchmarks, and creates a production-ready package.
    
    Args:
        model_path (str): Path to the trained PyTorch model checkpoint.
    
    Returns:
        tuple: (prod_model_path, metadata_path, benchmark_results) or
               (None, None, None) if optimization fails.
    """
    print_step("4", "Model Optimization")
    
    try:
        from inference_optimizer import InferenceOptimizer
        
        # Create optimizer
        optimizer = InferenceOptimizer(model_path)
        
        # Optimize to TorchScript
        optimizer.optimize_to_torchscript()
        
        # Optimize to ONNX
        optimizer.optimize_to_onnx()
        
        # Run benchmarks
        print("Running benchmark...")
        results = optimizer.benchmark_models(num_runs=50)
        
        # Create production model
        prod_model_path, metadata_path = optimizer.create_production_model('torchscript')
        
        print("Optimization completed")
        return prod_model_path, metadata_path, results
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        return None, None, None

def setup_api_server():
    """
    Configure the API server for model inference.
    
    Verifies that optimized models exist and provides instructions
    for starting the API server.
    
    Returns:
        bool: True if configuration successful, False otherwise.
    """
    print_step("5", "API Server Configuration")
    
    try:
        # Verify optimized model directory exists
        model_dir = Path("./optimized_models")
        if not model_dir.exists():
            print("Warning: Optimized models directory not found")
            return False
        
        model_files = list(model_dir.glob("production_model.*"))
        if not model_files:
            print("Warning: No optimized model found")
            return False
        
        print("API server configured")
        print(f"   Model found: {model_files[0]}")
        print(f"   To start API: python api_server.py")
        print(f"   URL: http://localhost:8000")
        print(f"   Docs: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"Error configuring API: {e}")
        return False

def install_dependencies():
    """
    Install required Python dependencies.
    
    Installs all packages required for training, including PyTorch with ROCm
    support for AMD GPUs, image processing libraries, and API frameworks.
    """
    print_header("DEPENDENCY INSTALLATION")
    
    # List of dependencies
    dependencies = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2",  # ROCm for AMD
        "opencv-python",
        "albumentations",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "Pillow",
        "tqdm",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "aiofiles",
        "onnx",
        "onnxruntime"
    ]
    
    print("Installing dependencies...")
    print("   This may take several minutes...")
    
    for dep in dependencies:
        print(f"   Installing: {dep}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dep.split(), 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"   Warning: Error installing {dep}: {e}")
    
    print("Dependency installation completed")

def create_project_structure():
    """
    Create the project directory structure.
    
    Creates all necessary directories for storing models, processed data,
    uploads, temporary files, and logs.
    """
    directories = [
        "models",
        "optimized_models", 
        "processed_data",
        "uploads",
        "temp",
        "logs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("Directory structure created")

def main():
    """
    Main entry point for the training pipeline.
    
    Parses command line arguments and executes the complete training
    pipeline including data analysis, preprocessing, training,
    optimization, and API configuration.
    """
    parser = argparse.ArgumentParser(description="Complete pipeline for DOG vs NON-DOG classification")
    parser.add_argument("--dataset", required=True, help="Path to the DATASETS directory")
    parser.add_argument("--model", default="efficientnet_b3", help="Model architecture (efficientnet_b3, resnet50, etc.)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--balance", default="undersample", help="Balancing strategy (undersample, oversample, none)")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip data analysis")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (use existing model)")
    parser.add_argument("--api-only", action="store_true", help="Only configure API")
    
    args = parser.parse_args()
    
    print_header("DOG CLASSIFICATION - COMPLETE PIPELINE")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Balance: {args.balance}")
    
    # Verify dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Create project structure
    create_project_structure()
    
    # Install dependencies
    if not args.skip_install:
        install_dependencies()
    
    if args.api_only:
        setup_api_server()
        return
    
    # Step 1: Data Analysis
    if not args.skip_analysis:
        success = run_data_analysis(str(dataset_path))
        if not success:
            print("Error in data analysis. Aborting.")
            return
    
    # Step 2: Preprocessing
    output_path = "./processed_data"
    data_loaders, splits = run_data_preprocessing(
        str(dataset_path), 
        output_path, 
        args.balance
    )
    
    if data_loaders is None:
        print("Error in preprocessing. Aborting.")
        return
    
    # Step 3: Training
    if not args.skip_training:
        trainer, model_path = run_model_training(
            data_loaders, 
            args.model, 
            args.epochs
        )
        
        if trainer is None:
            print("Error in training. Aborting.")
            return
    else:
        # Search for existing model
        model_path = "./models/best_model.pth"
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            return
    
    # Step 4: Optimization
    prod_model_path, metadata_path, benchmark_results = run_model_optimization(model_path)
    
    if prod_model_path is None:
        print("Error in optimization. Aborting.")
        return
    
    # Step 5: Configure API
    api_success = setup_api_server()
    
    # Final Summary
    print_header("FINAL SUMMARY")
    print("Pipeline completed successfully!")
    print(f"Original model: {model_path}")
    print(f"Optimized model: {prod_model_path}")
    print(f"Metadata: {metadata_path}")
    
    if benchmark_results:
        print("\nBEST BENCHMARK RESULTS:")
        fastest_key = min(benchmark_results.keys(), key=lambda k: benchmark_results[k]['avg_time_ms'])
        print(f"   {fastest_key}: {benchmark_results[fastest_key]['avg_time_ms']:.2f} ms, {benchmark_results[fastest_key]['fps']:.1f} FPS")
    
    print("\nAPI SERVER:")
    print("   To start: python api_server.py")
    print("   URL: http://localhost:8000")
    print("   Documentation: http://localhost:8000/docs")
    
    print("\nQUICK TEST:")
    print("   curl -X POST 'http://localhost:8000/predict' -F 'file=@image.jpg'")

if __name__ == "__main__":
    main()