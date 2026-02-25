# !/usr/bin/env python3
"""
Detailed Class Balance Analysis Module.

This module provides comprehensive analysis of class distribution imbalance
in the breed classification dataset. It calculates statistical metrics,
identifies outliers, and proposes data balancing strategies.

Key Features:
    - Statistical analysis (mean, std, CV, quartiles)
    - Imbalance severity classification
    - Outlier detection using IQR method
    - Balancing strategy recommendations
    - Data augmentation/reduction planning

Output:
    - detailed_balance_report.json: Complete analysis results

Usage:
    python detailed_balance_analysis.py

Author: System IA
Date: 2024
"""

import os
import numpy as np
import json
from collections import Counter


def detailed_balance_analysis():
    """
    Perform comprehensive class balance analysis on the training dataset.

    This function analyzes the distribution of images across all breed classes
    and provides detailed statistics to assess the severity of class imbalance.

    Returns:
        dict: Analysis results containing:
            - breed_counts (dict): Image count per breed
            - stats (dict): Statistical metrics (mean, std, cv, quartiles)
            - outliers_low (list): Breeds with unusually few images
            - outliers_high (list): Breeds with unusually many images
            - balance_status (str): Severity classification

    Statistical Metrics Computed:
        - Mean, standard deviation, coefficient of variation
        - Min/max image counts
        - Quartiles (Q1, Q2/median, Q3)
        - IQR-based outlier detection

    Balance Status Categories:
        - CRITICAL: CV > 0.5 (severely imbalanced)
        - HIGH: CV > 0.3 (strongly imbalanced)
        - MEDIUM: CV > 0.1 (moderately imbalanced)
        - LOW: CV <= 0.1 (well balanced)
    """
    
    print("🔍 DETAILED CLASS BALANCE ANALYSIS")
    print("=" * 60)
    
    train_dir = "breed_processed_data/train"
    breeds = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    # Count images per breed
    breed_counts = {}
    for breed in breeds:
        breed_path = os.path.join(train_dir, breed)
        count = len([f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        breed_counts[breed] = count
    
    # Calculate statistical metrics
    counts = list(breed_counts.values())
    total_images = sum(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    min_count = min(counts)
    max_count = max(counts)
    cv = std_count / mean_count
    
    print(f"📊 GENERAL STATISTICS:")
    print(f"   Total breeds: {len(breeds)}")
    print(f"   Total images: {total_images:,}")
    print(f"   Average per breed: {mean_count:.1f}")
    print(f"   Standard deviation: {std_count:.1f}")
    print(f"   Coefficient of variation: {cv:.3f}")
    print(f"   Range: {min_count} - {max_count} ({max_count - min_count} difference)")
    
    # Classify imbalance severity
    if cv > 0.5:
        balance_status = "🔴 SEVERELY IMBALANCED"
        priority = "CRITICAL"
    elif cv > 0.3:
        balance_status = "🟠 STRONGLY IMBALANCED"
        priority = "HIGH"
    elif cv > 0.1:
        balance_status = "🟡 MODERATELY IMBALANCED"
        priority = "MEDIUM"
    else:
        balance_status = "🟢 WELL BALANCED"
        priority = "LOW"
    
    print(f"\n⚖️ BALANCE EVALUATION:")
    print(f"   Status: {balance_status}")
    print(f"   Correction priority: {priority}")
    
    # Sort breeds by image count
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1])
    
    print(f"\n📉 TOP 10 BREEDS WITH FEWEST IMAGES:")
    for i, (breed, count) in enumerate(sorted_breeds[:10], 1):
        deficit = mean_count - count
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>3} images ({percentage:.1f}%, deficit: {deficit:+.0f})")
    
    print(f"\n📈 TOP 10 BREEDS WITH MOST IMAGES:")
    for i, (breed, count) in enumerate(sorted_breeds[-10:], 1):
        excess = count - mean_count
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>3} images ({percentage:.1f}%, excess: {excess:+.0f})")
    
    # Quartile analysis
    q1 = np.percentile(counts, 25)
    q2 = np.percentile(counts, 50)  # median
    q3 = np.percentile(counts, 75)
    iqr = q3 - q1
    
    print(f"\n📊 QUARTILE ANALYSIS:")
    print(f"   Q1 (25%): {q1:.1f}")
    print(f"   Q2 (50%, median): {q2:.1f}")
    print(f"   Q3 (75%): {q3:.1f}")
    print(f"   IQR: {iqr:.1f}")
    
    # Detect outliers using IQR method
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr
    
    outliers_low = [(breed, count) for breed, count in breed_counts.items() if count < outlier_threshold_low]
    outliers_high = [(breed, count) for breed, count in breed_counts.items() if count > outlier_threshold_high]
    
    if outliers_low or outliers_high:
        print(f"\n🚨 OUTLIERS DETECTED:")
        if outliers_low:
            print(f"   Breeds with too few images (< {outlier_threshold_low:.1f}):")
            for breed, count in sorted(outliers_low, key=lambda x: x[1]):
                print(f"      - {breed}: {count}")
        
        if outliers_high:
            print(f"   Breeds with too many images (> {outlier_threshold_high:.1f}):")
            for breed, count in sorted(outliers_high, key=lambda x: x[1], reverse=True):
                print(f"      - {breed}: {count}")
    
    return {
        'breed_counts': breed_counts,
        'stats': {
            'total_breeds': len(breeds),
            'total_images': total_images,
            'mean': mean_count,
            'std': std_count,
            'cv': cv,
            'min': min_count,
            'max': max_count,
            'q1': q1,
            'q2': q2,
            'q3': q3
        },
        'outliers_low': outliers_low,
        'outliers_high': outliers_high,
        'balance_status': priority
    }

def propose_balancing_strategy(analysis):
    """
    Propose data balancing strategy based on analysis results.

    This function analyzes the class distribution statistics and proposes
    appropriate balancing techniques including data augmentation for
    under-represented classes and reduction strategies for over-represented ones.

    Args:
        analysis (dict): Results from detailed_balance_analysis() containing
            breed_counts, stats, and outlier information.

    Returns:
        dict: Balancing strategy containing:
            - target_count (int): Target images per class
            - strategy (str): 'AGGRESSIVE' or 'CONSERVATIVE'
            - breeds_to_augment (list): Classes needing more images
            - breeds_to_reduce (list): Classes needing fewer images
            - breeds_balanced (list): Already balanced classes
            - total_augmentation_needed (int): Total images to generate
            - total_reduction_possible (int): Total images to remove

    Strategy Selection:
        - AGGRESSIVE: For CV > 0.3, uses median as target
        - CONSERVATIVE: For CV <= 0.3, uses mean as target
    """
    
    print(f"\n🎯 RECOMMENDED BALANCING STRATEGY")
    print("=" * 60)
    
    stats = analysis['stats']
    cv = stats['cv']
    mean_count = stats['mean']
    
    # Define balancing target
    if cv > 0.3:
        # For highly imbalanced datasets, use median as target
        target_count = int(stats['q2'])
        strategy = "AGGRESSIVE"
    else:
        # For moderately imbalanced, use adjusted mean
        target_count = int(mean_count)
        strategy = "CONSERVATIVE"
    
    print(f"📋 BALANCING PARAMETERS:")
    print(f"   Strategy: {strategy}")
    print(f"   Target per breed: {target_count} images")
    print(f"   Acceptable range: {int(target_count * 0.9)} - {int(target_count * 1.1)}")
    
    # Calculate required actions
    breeds_to_augment = []  # Need more images
    breeds_to_reduce = []   # Need fewer images
    breeds_balanced = []    # Already within acceptable range
    
    for breed, count in analysis['breed_counts'].items():
        if count < target_count * 0.9:
            needed = target_count - count
            breeds_to_augment.append((breed, count, needed))
        elif count > target_count * 1.1:
            excess = count - target_count
            breeds_to_reduce.append((breed, count, excess))
        else:
            breeds_balanced.append((breed, count))
    
    print(f"\n📈 BREEDS NEEDING AUGMENTATION ({len(breeds_to_augment)}):")
    total_augmentation_needed = 0
    for breed, current, needed in sorted(breeds_to_augment, key=lambda x: x[2], reverse=True):
        print(f"   {breed}: {current} → {target_count} (+{needed})")
        total_augmentation_needed += needed
    
    print(f"\n📉 BREEDS NEEDING REDUCTION ({len(breeds_to_reduce)}):")
    total_reduction_possible = 0
    for breed, current, excess in sorted(breeds_to_reduce, key=lambda x: x[2], reverse=True):
        print(f"   {breed}: {current} → {target_count} (-{excess})")
        total_reduction_possible += excess
    
    print(f"\n✅ ALREADY BALANCED BREEDS ({len(breeds_balanced)}):")
    for breed, count in breeds_balanced:
        print(f"   {breed}: {count} ✓")
    
    print(f"\n📊 ACTION SUMMARY:")
    print(f"   Total images to generate: {total_augmentation_needed}")
    print(f"   Total images to reduce: {total_reduction_possible}")
    print(f"   Net balance: {total_augmentation_needed - total_reduction_possible:+d}")
    
    # Technique recommendations
    print(f"\n🔧 RECOMMENDED TECHNIQUES:")
    
    if len(breeds_to_augment) > 0:
        print("   For image augmentation:")
        print("      - Data Augmentation (rotation, flip, zoom)")
        print("      - Synthetic generation with GANs")
        print("      - Supervised web scraping")
        print("      - Transfer learning from similar breeds")
    
    if len(breeds_to_reduce) > 0:
        print("   For image reduction:")
        print("      - Stratified random sampling")
        print("      - Keep only highest quality images")
        print("      - Preserve diversity within each breed")
    
    print("   Other recommendations:")
    print("      - Weighted loss function during training")
    print("      - Class-balanced sampling during training")
    print("      - Stratified validation set")
    
    return {
        'target_count': target_count,
        'strategy': strategy,
        'breeds_to_augment': breeds_to_augment,
        'breeds_to_reduce': breeds_to_reduce,
        'breeds_balanced': breeds_balanced,
        'total_augmentation_needed': total_augmentation_needed,
        'total_reduction_possible': total_reduction_possible
    }


if __name__ == "__main__":
    print("Running detailed balance analysis...")
    
    # Execute balance analysis
    analysis = detailed_balance_analysis()
    
    # Propose balancing strategy
    strategy = propose_balancing_strategy(analysis)
    
    # Save results
    results = {
        'analysis': analysis,
        'strategy': strategy
    }
    
    with open('detailed_balance_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: detailed_balance_report.json")