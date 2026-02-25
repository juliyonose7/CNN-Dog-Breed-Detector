#!/usr/bin/env python3
"""
Model Diagnostics and Testing Module.

This script provides comprehensive testing for the dog detection model.
It validates both direct model inference and API endpoint predictions,
diagnosing issues with model accuracy and generating test reports.

Key Features:
    - Direct model inference testing
    - API endpoint validation
    - Automatic test image discovery from YESDOG dataset
    - Comparative analysis of direct vs API predictions
    - Diagnostic recommendations for detected issues
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import io
import base64
from pathlib import Path
import json


class DogClassificationModel(nn.Module):
    """
    Binary dog classification model based on ResNet50.
    
    Recreates the exact architecture from quick_train.py for model loading.
    
    Attributes:
        backbone (nn.Module): ResNet50 feature extractor.
        classifier (nn.Sequential): Custom classification head.
    """
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1, pretrained: bool = True):
        """
        Initialize the classification model.
        
        Args:
            model_name (str): Backbone architecture. Default: 'resnet50'.
            num_classes (int): Number of output classes. Default: 1 (binary).
            pretrained (bool): Use ImageNet pretrained weights. Default: True.
        """
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
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Logit output, squeezed for binary classification.
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()


def load_model():
    """
    Load the trained binary classification model.
    
    Returns:
        tuple: (model, transform) if successful, (None, None) otherwise.
    """
    model_path = Path("./quick_models/best_model.pth")
    
    if not model_path.exists():
        print(" Model not found")
        return None, None
    
    model = DogClassificationModel(model_name='resnet50', num_classes=1, pretrained=False)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Transformations (same as training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(" Model loaded successfully")
        return model, transform
    except Exception as e:
        print(f" Error loading model: {e}")
        return None, None


def predict_image(model, transform, image_path):
    """
    Predict whether an image contains a dog.
    
    Args:
        model (nn.Module): Loaded classification model.
        transform: Image transformation pipeline.
        image_path (Path): Path to the image file.
        
    Returns:
        dict: Prediction results with is_dog, probability, confidence, and raw_output.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
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
    """
    Test the prediction API endpoint.
    
    Args:
        image_path (Path): Path to test image.
        
    Returns:
        dict: API response or error details.
    """
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
    """
    Discover and prepare test images from the YESDOG dataset.
    
    Searches for existing dog images in the dataset to use as test cases.
    
    Returns:
        list: List of Path objects to test images (max 5).
    """
    print(" Creating test directory...")
    test_dir = Path("./test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Use images from YESDOG dataset for testing
    print(" Using images from YESDOG dataset for tests...")
    
    dog_images = []
    yesdog_dir = Path("./DATASETS/YESDOG")
    
    if yesdog_dir.exists():
        # Find some dog images from the dataset
        for breed_dir in list(yesdog_dir.iterdir())[:3]:  # Only 3 breeds
            if breed_dir.is_dir():
                breed_images = list(breed_dir.glob("*.jpg"))[:2]  # 2 images per breed
                dog_images.extend(breed_images)
                if len(dog_images) >= 5:
                    break
    
    return dog_images[:5]  # Maximum 5 test images


def main():
    """
    Main diagnostic function.
    
    Loads the model, runs tests on sample images, and generates
    a diagnostic report with recommendations.
    """
    print(" Starting dog detection model diagnostics")
    print("=" * 60)
    
    # Load model
    model, transform = load_model()
    if not model:
        return
    
    # Get test images
    test_images = create_test_images()
    
    if not test_images:
        print(" No test images found")
        return
    
    print(f"  Testing with {len(test_images)} dog images...")
    print()
    
    results = []
    
    for i, image_path in enumerate(test_images, 1):
        print(f" Image {i}: {image_path.name}")
        
        # Direct model prediction
        direct_result = predict_image(model, transform, image_path)
        
        # API prediction
        api_result = test_api_endpoint(image_path)
        
        results.append({
            'image': image_path.name,
            'direct': direct_result,
            'api': api_result
        })
        
        print(f"    Direct model: {' DOG' if direct_result['is_dog'] else ' NOT-DOG'} "
              f"(prob: {direct_result['probability']:.3f})")
        
        if 'error' not in api_result:
            api_is_dog = api_result.get('class') == 'dog'
            api_confidence = api_result.get('confidence', 0)
            print(f"    API: {' DOG' if api_is_dog else ' NOT-DOG'} "
                  f"(conf: {api_confidence:.3f})")
        else:
            print(f"    API:  Error - {api_result['error']}")
        
        print()
    
    # Summary
    print(" SUMMARY:")
    direct_correct = sum(1 for r in results if r['direct']['is_dog'])
    api_correct = sum(1 for r in results if 'error' not in r['api'] and r['api'].get('class') == 'dog')
    
    print(f"   Direct model: {direct_correct}/{len(results)} dogs detected")
    print(f"   API: {api_correct}/{len(results)} dogs detected")
    
    if direct_correct == 0:
        print("  PROBLEM: The model is not detecting dogs correctly")
        print("   Possible causes:")
        print("   1. Model poorly trained")
        print("   2. Incorrect transformations")
        print("   3. Decision threshold too high")
        print()
        print(" Recommended solutions:")
        print("   1. Retrain with more epochs")
        print("   2. Verify training dataset")
        print("   3. Adjust threshold from 0.5 to 0.3")
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f" Results saved to: test_results.json")

if __name__ == "__main__":
    main()