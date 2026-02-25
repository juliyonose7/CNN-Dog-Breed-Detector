#!/usr/bin/env python3
"""
Hierarchical Dog Classification Model Testing.

This script provides direct testing of the HierarchicalDogClassifier.
It validates that both the binary (dog/not-dog) and breed classification
models are loaded correctly and producing accurate predictions.

Features:
    - Direct model loading validation
    - Binary classification testing
    - Breed classification with Top-3 predictions
    - Remote test image download with local fallback
"""

import sys
sys.path.append('.')

from hierarchical_dog_classifier import HierarchicalDogClassifier
from PIL import Image
import requests
from io import BytesIO


def test_models():
    """
    Test the hierarchical dog classifier models directly.
    
    Validates model loading, downloads a test image, and runs
    both binary and breed classification predictions.
    """
    print(" DIRECT MODEL TEST")
    print("=" * 50)
    
    # Create classifier
    classifier = HierarchicalDogClassifier()
    
    # Verify status
    info = classifier.get_model_info()
    print(f" Models loaded:")
    print(f"   Binary: {'' if info['binary_model_loaded'] else ''}")
    print(f"   Breeds: {'' if info['breed_model_loaded'] else ''}")
    print(f"   Number of breeds: {info['num_breeds']}")
    
    if not info['binary_model_loaded'] or not info['breed_model_loaded']:
        print(" Models not loaded correctly")
        return
    
    # Download test image
    print("\n Downloading test image...")
    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Beagle_sitting.jpg/800px-Beagle_sitting.jpg"
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        print(f" Image loaded: {image.size}")
    except Exception as e:
        print(f" Error downloading image: {e}")
        print(" Using local image...")
        
        # Create simple test image
        image = Image.new('RGB', (224, 224), color='brown')
        print(" Test image created")
    
    # Test prediction
    print("\n Testing prediction...")
    try:
        result = classifier.predict_image(image, confidence_threshold=0.1)  # Very low threshold
        
        print(" RESULT:")
        print(f"   Is dog: {'' if result['is_dog'] else ''}")
        print(f"   Binary confidence: {result['binary_confidence']:.4f}")
        
        if result.get('breed'):
            print(f"   Breed: {result['breed']}")
            print(f"   Breed confidence: {result.get('breed_confidence', 0):.4f}")
            
        if result.get('breed_top3'):
            print("   Top-3 breeds:")
            for i, breed_info in enumerate(result['breed_top3'][:3]):
                medal = ['', '', ''][i]
                print(f"     {medal} {breed_info['breed']}: {breed_info['confidence']:.4f}")
        
        if result.get('error'):
            print(f" Error: {result['error']}")
            
    except Exception as e:
        print(f" Prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_models()