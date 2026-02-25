# !/usr/bin/env python3
"""
Script de prueba for verificar that los models funcionan correctamente
"""

import sys
sys.path.append('.')

from hierarchical_dog_classifier import HierarchicalDogClassifier
from PIL import Image
import requests
from io import BytesIO

def test_models():
    print("🧪 PRUEBA DE MODELOS DIRECTA")
    print("=" * 50)
    
    # Crear clasificador
    classifier = HierarchicalDogClassifier()
    
    # Verificar estado
    info = classifier.get_model_info()
    print(f"📊 Modelos cargados:")
    print(f"   Binary: {'✅' if info['binary_model_loaded'] else '❌'}")
    print(f"   Breeds: {'✅' if info['breed_model_loaded'] else '❌'}")
    print(f"   Razas: {info['num_breeds']}")
    
    if not info['binary_model_loaded'] or not info['breed_model_loaded']:
        print("❌ Modelos no cargados correctamente")
        return
    
    # Descargar image de prueba
    print("\n🖼️ Descargando imagen de prueba...")
    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Beagle_sitting.jpg/800px-Beagle_sitting.jpg"
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        print(f"✅ Imagen cargada: {image.size}")
    except Exception as e:
        print(f"❌ Error descargando imagen: {e}")
        print("📁 Usando imagen local...")
        
        # Crear image de prueba simple
        image = Image.new('RGB', (224, 224), color='brown')
        print("✅ Imagen de prueba creada")
    
    # Probar prediction
    print("\n🤖 Probando predicción...")
    try:
        result = classifier.predict_image(image, confidence_threshold=0.1)  # Threshold muy bajo
        
        print("📊 RESULTADO:")
        print(f"   Es perro: {'✅' if result['is_dog'] else '❌'}")
        print(f"   Confianza binaria: {result['binary_confidence']:.4f}")
        
        if result.get('breed'):
            print(f"   Raza: {result['breed']}")
            print(f"   Confianza raza: {result.get('breed_confidence', 0):.4f}")
            
        if result.get('breed_top3'):
            print("   Top-3 razas:")
            for i, breed_info in enumerate(result['breed_top3'][:3]):
                medal = ['🥇', '🥈', '🥉'][i]
                print(f"     {medal} {breed_info['breed']}: {breed_info['confidence']:.4f}")
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_models()