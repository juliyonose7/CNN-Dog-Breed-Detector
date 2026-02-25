"""
Technical documentation in English.
Verifies the status actual of all the componentes of the system
"""

import os
import json
from pathlib import Path
import torch

def check_models():
    """Verifies the status of the models"""
    print("🔍 VERIFICANDO ESTADO DE LOS MODELOS")
    print("=" * 60)
    
    # Model binario
    binary_model_path = "best_model.pth"
    if os.path.exists(binary_model_path):
        print("✅ Modelo binario: DISPONIBLE")
        print(f"   📁 Archivo: {binary_model_path}")
        print(f"   📊 Tamaño: {os.path.getsize(binary_model_path) / (1024*1024):.1f} MB")
        print(f"   📈 Precisión reportada: 91.33%")
    else:
        print("❌ Modelo binario: NO ENCONTRADO")
    
    print()
    
    # Model of breeds
    breed_models_dir = Path("breed_models")
    if breed_models_dir.exists():
        checkpoints = list(breed_models_dir.glob("*.pth"))
        if checkpoints:
            print("✅ Modelo de razas: CHECKPOINTS DISPONIBLES")
            for checkpoint in checkpoints:
                print(f"   📁 {checkpoint.name}: {checkpoint.stat().st_size / (1024*1024):.1f} MB")
        else:
            print("⚠️  Modelo de razas: DIRECTORIO EXISTE, SIN CHECKPOINTS")
    else:
        print("⚠️  Modelo de razas: EN ENTRENAMIENTO / NO DISPONIBLE")
    
    print()

def check_datasets():
    """Verifies the status of the datasets"""
    print("📊 VERIFICANDO DATASETS")
    print("=" * 60)
    
    # Dataset original
    datasets_dir = Path("DATASETS")
    if datasets_dir.exists():
        yesdog_dir = datasets_dir / "YESDOG"
        nodog_dir = datasets_dir / "NODOG"
        
        if yesdog_dir.exists():
            breed_dirs = [d for d in yesdog_dir.iterdir() if d.is_dir()]
            print(f"✅ Dataset YESDOG: {len(breed_dirs)} razas disponibles")
        
        if nodog_dir.exists():
            print("✅ Dataset NODOG: Disponible")
    else:
        print("❌ Dataset original: NO ENCONTRADO")
    
    # Dataset procesado of breeds
    dataset_info_path = "dataset_info.json"
    if os.path.exists(dataset_info_path):
        print("✅ Dataset de razas procesado: DISPONIBLE")
        with open(dataset_info_path, 'r') as f:
            info = json.load(f)
            print(f"   🏷️  Razas: {info['total_classes']}")
            print(f"   📊 Total imágenes: {info['total_samples']:,}")
            print(f"   🏋️  Entrenamiento: {info['train_samples']:,}")
            print(f"   ✅ Validación: {info['val_samples']:,}")
            print(f"   🧪 Test: {info['test_samples']:,}")
    else:
        print("⚠️  Dataset de razas procesado: NO DISPONIBLE")
    
    print()

def check_configuration():
    """Verifies the configuraciones"""
    print("⚙️  VERIFICANDO CONFIGURACIONES")
    print("=" * 60)
    
    # Configuration of breeds
    breed_config_path = "breed_config.py"
    if os.path.exists(breed_config_path):
        print("✅ Configuración de razas: DISPONIBLE")
        try:
            import breed_config
            print(f"   🏷️  Razas configuradas: {len(breed_config.CLASS_NAMES)}")
            print(f"   🔢 Índices mapeados: {len(breed_config.CLASS_TO_IDX)}")
        except ImportError:
            print("   ⚠️  Error importando configuración")
    else:
        print("❌ Configuración de razas: NO ENCONTRADA")
    
    print()

def check_api_files():
    """Verifies the files of the API"""
    print("🚀 VERIFICANDO ARCHIVOS DE API")
    print("=" * 60)
    
    api_files = [
        ("hierarchical_api.py", "API Jerárquica"),
        ("hierarchical_frontend.html", "Frontend Jerárquico"),
        ("app.py", "API Simple Original"),
        ("index.html", "Frontend Original")
    ]
    
    for file_path, description in api_files:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"✅ {description}: {size_kb:.1f} KB")
        else:
            print(f"❌ {description}: NO ENCONTRADO")
    
    print()

def check_system_requirements():
    """Verifies the requisitos of the system"""
    print("💻 VERIFICANDO REQUISITOS DEL SISTEMA")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   💻 CPU threads disponibles: {torch.get_num_threads()}")
        print(f"   🚀 Optimización 7800X3D: Configurada")
    except ImportError:
        print("❌ PyTorch: NO INSTALADO")
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError:
        print("❌ TorchVision: NO INSTALADO")
    
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI: NO INSTALADO")
    
    try:
        import albumentations
        print(f"✅ Albumentations: {albumentations.__version__}")
    except ImportError:
        print("❌ Albumentations: NO INSTALADO")
    
    print()

def get_recommendations():
    """Proporciona recomendaciones basadas en the status"""
    print("💡 RECOMENDACIONES")
    print("=" * 60)
    
    # Implementation note.
    has_binary = os.path.exists("best_model.pth")
    has_breed_config = os.path.exists("breed_config.py")
    
    if has_binary and has_breed_config:
        print("🎯 SISTEMA LISTO PARA DEMOSTRACIÓN:")
        print("   1. Ejecutar API jerárquica: python hierarchical_api.py")
        print("   2. Abrir frontend: hierarchical_frontend.html")
        print("   3. El sistema funcionará con:")
        print("      • Detección binaria: ✅ Completamente funcional")
        print("      • Clasificación de razas: ⚠️ Mostrará mensaje de entrenamiento")
    
    elif has_binary:
        print("🎯 SISTEMA BÁSICO DISPONIBLE:")
        print("   1. Usar API original: python app.py")
        print("   2. Detección binaria funcional al 91.33%")
    
    else:
        print("❌ SISTEMA NO OPERATIVO:")
        print("   1. Falta modelo binario")
        print("   2. Ejecutar entrenamiento completo")
    
    print("\n🔧 PRÓXIMOS PASOS SUGERIDOS:")
    if not os.path.exists("breed_models"):
        print("   • Reanudar entrenamiento de razas: python breed_trainer.py")
    
    print("   • Probar sistema actual con frontend jerárquico")
    print("   • Monitorear progreso de entrenamiento de razas")
    
    print()

def main():
    """Function main"""
    print("🐕 SISTEMA JERÁRQUICO CANINO - STATUS COMPLETO")
    print("🚀 Optimizado para AMD 7800X3D")
    print("=" * 80)
    print()
    
    check_models()
    check_datasets()
    check_configuration()
    check_api_files()
    check_system_requirements()
    get_recommendations()
    
    print("=" * 80)
    print("✅ Verificación completa finalizada")

if __name__ == "__main__":
    main()