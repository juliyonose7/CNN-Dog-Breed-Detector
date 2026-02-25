#!/usr/bin/env python3
"""
Dataset Class Balance Analyzer for Dog Breed Classification
============================================================

This module analyzes the balance of classes (breed distribution) in the
dog breed classification dataset. It provides:

- Statistical analysis of image counts per breed
- Identification of class imbalance issues
- Visualization of breed distribution
- Recommendations for balancing strategies

The analysis is critical for ensuring the model doesn't develop bias
toward over-represented breeds while underperforming on rare breeds.

Author: Dog Breed Classifier Team
Date: 2024
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json

def analyze_breed_balance():
    """
    Analyze the class balance in the breed training dataset.
    
    Counts images per breed, calculates distribution statistics,
    and identifies potential class imbalance issues that could
    affect model training.
    
    Returns:
        dict: Analysis results containing breed counts, statistics,
              and balance status assessment.
    """
    
    print("🔍 CLASS BALANCE ANALYSIS - 50 BREEDS")
    print("=" * 60)
    
    train_dir = "breed_processed_data/train"
    if not os.path.exists(train_dir):
        print("❌ Training directory not found!")
        return
    
    # Count images per breed
    breed_counts = {}
    total_images = 0
    
    breeds = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    breeds.sort()
    
    print(f"📊 Found {len(breeds)} breeds:")
    print("-" * 60)
    
    for breed in breeds:
        breed_path = os.path.join(train_dir, breed)
        count = len([f for f in os.listdir(breed_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        breed_counts[breed] = count
        total_images += count
        print(f"   {breed}: {count:>4} images")
    
    print(f"\n📈 STATISTICS:")
    print(f"   Total images: {total_images:,}")
    print(f"   Average per breed: {total_images/len(breeds):.1f}")
    print(f"   Minimum: {min(breed_counts.values())} ({min(breed_counts, key=breed_counts.get)})")
    print(f"   Maximum: {max(breed_counts.values())} ({max(breed_counts, key=breed_counts.get)})")
    print(f"   Standard deviation: {np.std(list(breed_counts.values())):.1f}")
    
    # Calculate imbalance metrics
    counts = list(breed_counts.values())
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    cv = std_count / mean_count  # Coefficient of variation
    
    print(f"\n⚖️ IMBALANCE ANALYSIS:")
    print(f"   Coefficient of variation: {cv:.3f}")
    if cv > 0.3:
        print("   🔴 DATASET SIGNIFICANTLY IMBALANCED")
    elif cv > 0.1:
        print("   🟡 DATASET MODERATELY IMBALANCED") 
    else:
        print("   🟢 DATASET WELL BALANCED")
    
    # Show top/bottom breeds
    print(f"\n📊 TOP 10 BREEDS WITH MOST IMAGES:")
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (breed, count) in enumerate(sorted_breeds[:10], 1):
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>4} ({percentage:.1f}%)")
    
    print(f"\n📊 TOP 10 BREEDS WITH FEWEST IMAGES:")
    for i, (breed, count) in enumerate(sorted_breeds[-10:], 1):
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>4} ({percentage:.1f}%)")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    breeds_list = [breed.replace('_', ' ').title() for breed, _ in sorted_breeds]
    counts_list = [count for _, count in sorted_breeds]
    
    plt.bar(range(len(breeds_list)), counts_list, color='skyblue', alpha=0.7)
    plt.axhline(y=mean_count, color='red', linestyle='--', label=f'Average: {mean_count:.1f}')
    plt.axhline(y=mean_count + std_count, color='orange', linestyle=':', alpha=0.7, label=f'+1 SD: {mean_count + std_count:.1f}')
    plt.axhline(y=mean_count - std_count, color='orange', linestyle=':', alpha=0.7, label=f'-1 SD: {mean_count - std_count:.1f}')
    
    plt.xlabel('Breeds (sorted by count)')
    plt.ylabel('Number of images')
    plt.title('Image Distribution by Dog Breed (50 breeds)')
    plt.xticks(range(0, len(breeds_list), 5), 
               [breeds_list[i][:10] + '...' if len(breeds_list[i]) > 10 else breeds_list[i] 
                for i in range(0, len(breeds_list), 5)], 
               rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('breed_balance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save resultados
    results = {
        'total_breeds': len(breeds),
        'total_images': total_images,
        'mean_images_per_breed': mean_count,
        'std_deviation': std_count,
        'coefficient_of_variation': cv,
        'min_images': min(counts),
        'max_images': max(counts),
        'breed_counts': breed_counts,
        'balance_status': 'imbalanced' if cv > 0.1 else 'balanced'
    }
    
    with open('breed_balance_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   - breed_balance_report.json")
    print(f"   - breed_balance_analysis.png")
    
    return results

def recommend_balancing_strategy(results):
    """
    Recommend a balancing strategy based on analysis results.
    
    Analyzes the coefficient of variation and provides specific
    recommendations for data augmentation, undersampling, or
    weighted loss functions.
    
    Args:
        results (dict): Results from analyze_breed_balance() containing
                       breed counts and statistics.
    
    Returns:
        dict: Balancing recommendations including target per class,
              breeds needing augmentation, and strategies.
    """
    print(f"\n🎯 BALANCING RECOMMENDATIONS:")
    
    cv = results['coefficient_of_variation']
    mean_count = results['mean_images_per_breed']
    
    if cv > 0.3:
        print("   🔴 Dataset heavily imbalanced - Aggressive balancing required:")
        print("      - Data augmentation for breeds with few images")
        print("      - Undersampling for breeds with too many images")
        print("      - Weighted loss function during training")
        target_per_class = min(int(mean_count * 1.2), max(results['breed_counts'].values()))
    elif cv > 0.1:
        print("   🟡 Dataset moderately imbalanced - Light balancing:")
        print("      - Light data augmentation")
        print("      - Class weights in loss function")
        target_per_class = int(mean_count)
    else:
        print("   🟢 Dataset well balanced - Maintain as is")
        return results
    
    print(f"   📊 Recommended target: {target_per_class} images per breed")
    
    # Calculate balancing needs
    breeds_need_more = []
    breeds_need_less = []
    
    for breed, count in results['breed_counts'].items():
        if count < target_per_class * 0.8:  # Less than 80% of target
            breeds_need_more.append((breed, count, target_per_class - count))
        elif count > target_per_class * 1.5:  # More than 150% of target
            breeds_need_less.append((breed, count, count - target_per_class))
    
    if breeds_need_more:
        print(f"\n📈 Breeds needing MORE images ({len(breeds_need_more)}):")
        for breed, current, needed in sorted(breeds_need_more, key=lambda x: x[2], reverse=True)[:10]:
            print(f"      {breed}: {current} → {target_per_class} (+{needed})")
    
    if breeds_need_less:
        print(f"\n📉 Breeds needing FEWER images ({len(breeds_need_less)}):")
        for breed, current, excess in sorted(breeds_need_less, key=lambda x: x[2], reverse=True)[:10]:
            print(f"      {breed}: {current} → {target_per_class} (-{excess})")
    
    return {
        'target_per_class': target_per_class,
        'breeds_need_more': breeds_need_more,
        'breeds_need_less': breeds_need_less,
        'balancing_required': len(breeds_need_more) > 0 or len(breeds_need_less) > 0
    }

if __name__ == "__main__":
    # Ensure the main system is not executed
    import sys
    sys.path.insert(0, '.')
    
    results = analyze_breed_balance()
    if results:
        balancing = recommend_balancing_strategy(results)
        
        if balancing and balancing.get('balancing_required'):
            print(f"\n🔧 Do you want to proceed with automatic balancing? (y/n)")
            # For automation, assume 'y'
            response = 'y'
            if response.lower() == 'y':
                print("✅ Proceeding with automatic balancing...")
            else:
                print("⏹️ Balancing cancelled by user")