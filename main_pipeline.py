"""
Script principal for el training completo of the model PERRO vs NO-PERRO
Technical documentation in English.
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import time
import json

def print_header(title: str):
    """Imprime un header formateado"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step: str, description: str):
    """Imprime un paso of the process"""
    print(f"\n📋 PASO: {step}")
    print(f"   {description}")

def run_data_analysis(dataset_path: str):
    """Technical documentation in English."""
    print_step("1", "Análisis de Dataset")
    
    try:
        from data_analyzer import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer(dataset_path)
        analyzer.run_complete_analysis()
        
        print("✅ Análisis de datos completado")
        return True
        
    except Exception as e:
        print(f"❌ Error en análisis de datos: {e}")
        return False

def run_data_preprocessing(dataset_path: str, output_path: str, balance_strategy: str = 'undersample'):
    """Ejecuta el preprocesamiento de data"""
    print_step("2", "Preprocesamiento de Datos")
    
    try:
        from data_preprocessor import DataPreprocessor, create_sample_visualization
        
        preprocessor = DataPreprocessor(dataset_path, output_path)
        data_loaders, splits = preprocessor.process_complete_dataset(
            balance_strategy=balance_strategy,
            batch_size=32
        )
        
        # Implementation note.
        sample_viz_path = Path(output_path) / 'sample_visualization.png'
        create_sample_visualization(data_loaders, str(sample_viz_path))
        
        print("✅ Preprocesamiento de datos completado")
        return data_loaders, splits
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return None, None

def run_model_training(data_loaders, model_name: str = 'efficientnet_b3', num_epochs: int = 30):
    """Ejecuta el training of the model"""
    print_step("3", f"Entrenamiento del Modelo ({model_name})")
    
    try:
        from model_trainer import ModelTrainer, setup_rocm_optimization
        
        # Configurar ROCm
        rocm_available = setup_rocm_optimization()
        
        # Crear trainer
        trainer = ModelTrainer(model_name=model_name)
        
        # Configurar training
        trainer.setup_training(data_loaders['train'], data_loaders['val'])
        
        # Entrenar model
        models_dir = Path('./models')
        models_dir.mkdir(exist_ok=True)
        
        history = trainer.train_model(
            num_epochs=num_epochs,
            save_path=str(models_dir),
            freeze_epochs=5
        )
        
        print("✅ Entrenamiento completado")
        return trainer, str(models_dir / 'best_model.pth')
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return None, None

def run_model_optimization(model_path: str):
    """Ejecuta la optimization of the model"""
    print_step("4", "Optimización del Modelo")
    
    try:
        from inference_optimizer import InferenceOptimizer
        
        # Crear optimizador
        optimizer = InferenceOptimizer(model_path)
        
        # Optimizar a TorchScript
        optimizer.optimize_to_torchscript()
        
        # Optimizar a ONNX
        optimizer.optimize_to_onnx()``
        
        # Benchmark
        print("⏱️  Ejecutando benchmark...")
        results = optimizer.benchmark_models(num_runs=50)
        
        # Implementation note.
        prod_model_path, metadata_path = optimizer.create_production_model('torchscript')
        
        print("✅ Optimización completada")
        return prod_model_path, metadata_path, results
        
    except Exception as e:
        print(f"❌ Error en optimización: {e}")
        return None, None, None

def setup_api_server():
    """Configura el server API"""
    print_step("5", "Configuración del Servidor API")
    
    try:
        # Verificar that el model optimized existe
        model_dir = Path("./optimized_models")
        if not model_dir.exists():
            print("⚠️  Directorio de modelos optimizados no encontrado")
            return False
        
        model_files = list(model_dir.glob("production_model.*"))
        if not model_files:
            print("⚠️  No se encontró modelo optimizado")
            return False
        
        print("✅ Servidor API configurado")
        print(f"   Modelo encontrado: {model_files[0]}")
        print(f"   Para iniciar API: python api_server.py")
        print(f"   URL: http://localhost:8000")
        print(f"   Docs: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configurando API: {e}")
        return False

def install_dependencies():
    """Instala las dependencias necesarias"""
    print_header("INSTALACIÓN DE DEPENDENCIAS")
    
    # List de dependencias
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
    
    print("📦 Instalando dependencias...")
    print("   Esto puede tomar varios minutos...")
    
    for dep in dependencies:
        print(f"   Instalando: {dep}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dep.split(), 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Error instalando {dep}: {e}")
    
    print("✅ Instalación de dependencias completada")

def create_project_structure():
    """Crea la estructura de directorios of the proyecto"""
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
    
    print("📁 Estructura de directorios creada")

def main():
    """Function principal"""
    parser = argparse.ArgumentParser(description="Pipeline completo para clasificación PERRO vs NO-PERRO")
    parser.add_argument("--dataset", required=True, help="Ruta al directorio DATASETS")
    parser.add_argument("--model", default="efficientnet_b3", help="Modelo a usar (efficientnet_b3, resnet50, etc.)")
    parser.add_argument("--epochs", type=int, default=30, help="Número de épocas de entrenamiento")
    parser.add_argument("--balance", default="undersample", help="Estrategia de balanceo (undersample, oversample, none)")
    parser.add_argument("--skip-install", action="store_true", help="Saltar instalación de dependencias")
    parser.add_argument("--skip-analysis", action="store_true", help="Saltar análisis de datos")
    parser.add_argument("--skip-training", action="store_true", help="Saltar entrenamiento (usar modelo existente)")
    parser.add_argument("--api-only", action="store_true", help="Solo configurar API")
    
    args = parser.parse_args()
    
    print_header("DOG CLASSIFICATION - PIPELINE COMPLETO")
    print(f"🎯 Dataset: {args.dataset}")
    print(f"🤖 Modelo: {args.model}")
    print(f"📊 Épocas: {args.epochs}")
    print(f"⚖️ Balance: {args.balance}")
    
    # Verificar that el dataset existe
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset no encontrado en {dataset_path}")
        return
    
    # Crear estructura of the proyecto
    create_project_structure()
    
    # Instalar dependencias
    if not args.skip_install:
        install_dependencies()
    
    if args.api_only:
        setup_api_server()
        return
    
    # Implementation note.
    if not args.skip_analysis:
        success = run_data_analysis(str(dataset_path))
        if not success:
            print("❌ Error en análisis de datos. Abortando.")
            return
    
    # 2. Preprocesamiento
    output_path = "./processed_data"
    data_loaders, splits = run_data_preprocessing(
        str(dataset_path), 
        output_path, 
        args.balance
    )
    
    if data_loaders is None:
        print("❌ Error en preprocesamiento. Abortando.")
        return
    
    # 3. Training
    if not args.skip_training:
        trainer, model_path = run_model_training(
            data_loaders, 
            args.model, 
            args.epochs
        )
        
        if trainer is None:
            print("❌ Error en entrenamiento. Abortando.")
            return
    else:
        # Buscar model existente
        model_path = "./models/best_model.pth"
        if not Path(model_path).exists():
            print(f"❌ Modelo no encontrado: {model_path}")
            return
    
    # 4. Optimization
    prod_model_path, metadata_path, benchmark_results = run_model_optimization(model_path)
    
    if prod_model_path is None:
        print("❌ Error en optimización. Abortando.")
        return
    
    # 5. Configurar API
    api_success = setup_api_server()
    
    # Resumen final
    print_header("RESUMEN FINAL")
    print("✅ Pipeline completado exitosamente!")
    print(f"📁 Modelo original: {model_path}")
    print(f"🚀 Modelo optimizado: {prod_model_path}")
    print(f"📋 Metadata: {metadata_path}")
    
    if benchmark_results:
        print("\n📊 MEJORES RESULTADOS DE BENCHMARK:")
        fastest_key = min(benchmark_results.keys(), key=lambda k: benchmark_results[k]['avg_time_ms'])
        print(f"   {fastest_key}: {benchmark_results[fastest_key]['avg_time_ms']:.2f} ms, {benchmark_results[fastest_key]['fps']:.1f} FPS")
    
    print("\n🌐 SERVIDOR API:")
    print("   Para iniciar: python api_server.py")
    print("   URL: http://localhost:8000")
    print("   Documentación: http://localhost:8000/docs")
    
    print("\n🧪 PRUEBA RÁPIDA:")
    print("   curl -X POST 'http://localhost:8000/predict' -F 'file=@imagen.jpg'")

if __name__ == "__main__":
    main()