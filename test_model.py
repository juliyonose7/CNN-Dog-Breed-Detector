import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import io
import base64
from pathlib import Path
import json

# Recrear el model exactamente como en quick_train.py
class DogClassificationModel(nn.Module):
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1, pretrained: bool = True):
        super(DogClassificationModel, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()

def load_model():
    """Load el model entrenado"""
    model_path = Path("./quick_models/best_model.pth")
    
    if not model_path.exists():
        print("❌ Modelo no encontrado")
        return None, None
    
    model = DogClassificationModel(model_name='resnet50', num_classes=1, pretrained=False)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Transformaciones (mismas that en training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Modelo cargado exitosamente")
        return model, transform
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None

def predict_image(model, transform, image_path):
    """Predecir if una image contiene un perro"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Aplicar transformaciones
    input_tensor = transform(image).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
    
    is_dog = probability > 0.5
    return {
        'is_dog': is_dog,
        'probability': probability,
        'confidence': probability if is_dog else (1 - probability),
        'raw_output': output.item()
    }

def test_api_endpoint(image_path):
    """Probar el endpoint of the API"""
    url = "http://localhost:8000/predict"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'Status code: {response.status_code}', 'detail': response.text}
    except Exception as e:
        return {'error': str(e)}

def create_test_images():
    """Technical documentation in English."""
    print("📁 Creando directorio de pruebas...")
    test_dir = Path("./test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Implementation note.
    print("💡 Usando imágenes del dataset YESDOG para pruebas...")
    
    dog_images = []
    yesdog_dir = Path("./DATASETS/YESDOG")
    
    if yesdog_dir.exists():
        # Buscar algunas images de perros of the dataset
        for breed_dir in list(yesdog_dir.iterdir())[:3]:  # Only 3 breeds
            if breed_dir.is_dir():
                breed_images = list(breed_dir.glob("*.jpg"))[:2]  # 2 images por breed
                dog_images.extend(breed_images)
                if len(dog_images) >= 5:
                    break
    
    return dog_images[:5]  # Implementation note.

def main():
    print("🔍 Iniciando diagnóstico del modelo de detección de perros")
    print("=" * 60)
    
    # load model
    model, transform = load_model()
    if not model:
        return
    
    # Obtener images de prueba
    test_images = create_test_images()
    
    if not test_images:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    print(f"🖼️  Probando con {len(test_images)} imágenes de perros...")
    print()
    
    results = []
    
    for i, image_path in enumerate(test_images, 1):
        print(f"📷 Imagen {i}: {image_path.name}")
        
        # Prediction directa of the model
        direct_result = predict_image(model, transform, image_path)
        
        # Prediction via API
        api_result = test_api_endpoint(image_path)
        
        results.append({
            'image': image_path.name,
            'direct': direct_result,
            'api': api_result
        })
        
        print(f"   📊 Modelo directo: {'🐕 PERRO' if direct_result['is_dog'] else '❌ NO-PERRO'} "
              f"(prob: {direct_result['probability']:.3f})")
        
        if 'error' not in api_result:
            api_is_dog = api_result.get('class') == 'dog'
            api_confidence = api_result.get('confidence', 0)
            print(f"   🌐 API: {'🐕 PERRO' if api_is_dog else '❌ NO-PERRO'} "
                  f"(conf: {api_confidence:.3f})")
        else:
            print(f"   🌐 API: ❌ Error - {api_result['error']}")
        
        print()
    
    # Resumen
    print("📋 RESUMEN:")
    direct_correct = sum(1 for r in results if r['direct']['is_dog'])
    api_correct = sum(1 for r in results if 'error' not in r['api'] and r['api'].get('class') == 'dog')
    
    print(f"   Modelo directo: {direct_correct}/{len(results)} perros detectados")
    print(f"   API: {api_correct}/{len(results)} perros detectados")
    
    if direct_correct == 0:
        print("⚠️  PROBLEMA: El modelo no está detectando perros correctamente")
        print("   Posibles causas:")
        print("   1. Modelo mal entrenado")
        print("   2. Transformaciones incorrectas")
        print("   3. Umbral de decisión muy alto")
        print()
        print("💡 Soluciones recomendadas:")
        print("   1. Reentrenar con más épocas")
        print("   2. Verificar dataset de entrenamiento")
        print("   3. Ajustar umbral de 0.5 a 0.3")
    
    # Save resultados
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Resultados guardados en: test_results.json")

if __name__ == "__main__":
    main()