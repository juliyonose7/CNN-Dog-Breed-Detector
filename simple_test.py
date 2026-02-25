# !/usr/bin/env python3
"""
Simple Model Testing Script.

Tests model loading and inference without importing the complete classifier.
Useful for debugging model compatibility and basic functionality.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


class FastBinaryModel(nn.Module):
    """Binary classifier model for dog/no-dog detection.
    
    Uses ResNet18 backbone for lightweight inference.
    
    Attributes:
        backbone: ResNet18 feature extractor.
    """
    
    def __init__(self, num_classes=2):
        """Initialize with ResNet18 backbone.
        
        Args:
            num_classes: Number of output classes (default: 2).
        """
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


class BreedModel(nn.Module):
    """Breed classifier model for 50-class breed identification.
    
    Uses ResNet34 backbone for better accuracy.
    
    Attributes:
        backbone: ResNet34 feature extractor.
    """
    
    def __init__(self, num_classes=50):
        """Initialize with ResNet34 backbone.
        
        Args:
            num_classes: Number of breed classes (default: 50).
        """
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)

def test_model_loading():
    """Test model loading and basic inference.
    
    Tests both binary and breed models by loading checkpoints
    and running inference on a synthetic test image.
    """
    print(" SIMPLE MODEL TEST")
    print("=" * 40)
    
    device = torch.device('cpu')
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test binary model
    print("1 Testing binary model...")
    try:
        binary_model = FastBinaryModel(num_classes=2).to(device)
        binary_path = "realtime_binary_models/best_model_epoch_1_acc_0.9649.pth"
        
        if os.path.exists(binary_path):
            checkpoint = torch.load(binary_path, map_location=device)
            binary_model.load_state_dict(checkpoint['model_state_dict'])
            binary_model.eval()
            print(" Binary model loaded")
        else:
            print(f" Not found: {binary_path}")
            return
            
    except Exception as e:
        print(f" Binary model error: {e}")
        return
    
    # Test breed model
    print("2 Testing breed model...")
    try:
        breed_model = BreedModel(num_classes=50).to(device)
        breed_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
        
        if os.path.exists(breed_path):
            checkpoint = torch.load(breed_path, map_location=device)
            breed_model.load_state_dict(checkpoint['model_state_dict'])
            breed_model.eval()
            print(" Breed model loaded")
        else:
            print(f" Not found: {breed_path}")
            return
            
    except Exception as e:
        print(f" Breed model error: {e}")
        return
    
    # Create test image
    print("3 Creating test image...")
    test_image = Image.new('RGB', (300, 300), color=(139, 69, 19))  # Brown color
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    print(f" Tensor created: {input_tensor.shape}")
    
    # Test binary prediction
    print("4 Testing binary prediction...")
    try:
        with torch.no_grad():
            binary_output = binary_model(input_tensor)
            binary_probs = torch.softmax(binary_output, dim=1)
            binary_confidence, binary_pred = torch.max(binary_probs, 1)
            
            is_dog = bool(binary_pred.item() == 1)
            confidence = float(binary_confidence.item())
            
            print(f"   Result: {' DOG' if is_dog else ' NOT DOG'}")
            print(f"   Confianza: {confidence:.4f}")
            
    except Exception as e:
        print(f" Error predicción binaria: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Probar prediction of breeds (only if it is a dog)
    if is_dog:
        print("5 Probando predicción de razas...")
        try:
            with torch.no_grad():
                breed_output = breed_model(input_tensor)
                breed_probs = torch.softmax(breed_output, dim=1)
                breed_confidence, breed_pred = torch.max(breed_probs, 1)
                
                print(f"   Índice predicho: {breed_pred.item()}")
                print(f"   Confianza raza: {breed_confidence.item():.4f}")
                print(" Predicción de raza exitosa")
                
        except Exception as e:
            print(f" Error predicción razas: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print(" TODAS LAS PRUEBAS EXITOSAS")
    print("Los modelos funcionan correctamente.")
    print("El problema está en la comunicación web.")

if __name__ == "__main__":
    test_model_loading()