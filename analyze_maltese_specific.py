#!/usr/bin/env python3
"""
Maltese Dog Breed Specific Performance Analysis
================================================

This module performs detailed performance analysis specifically for the
Maltese dog breed. It provides:

- Individual breed performance metrics
- Confidence analysis and variance assessment
- Comparison with similar small dog breeds
- Percentile ranking among all breeds
- Bias detection specific to the Maltese class

This type of breed-specific analysis is useful for debugging individual
class performance issues and validating model behavior.

Author: Dog Breed Classifier Team
Date: 2024
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_maltese_performance():
    """
    Perform detailed performance analysis for the Maltese dog breed.
    
    Loads class metrics from JSON files and provides comprehensive
    analysis including accuracy, precision, recall, confidence statistics,
    and comparison with other similar breeds.
    
    Returns:
        dict: Analysis results containing performance level, bias assessment,
              metrics, and percentile rankings.
    """
    
    print(" SPECIFIC ANALYSIS: MALTESE DOG")
    print("=" * 50)
    
    # Load required data files
    try:
        with open('class_metrics.json', 'r') as f:
            class_metrics = json.load(f)
        
        with open('complete_class_evaluation_report.json', 'r') as f:
            complete_metrics = json.load(f)
            
    except Exception as e:
        print(f" Error loading data: {e}")
        return
    
    # Get Maltese-specific metrics
    maltese_metrics = class_metrics.get('Maltese_dog', {})
    maltese_complete = complete_metrics.get('class_reports', {}).get('Maltese_dog', {})
    
    if not maltese_metrics:
        print(" No metrics found for Maltese_dog")
        return
    
    print("\n PERFORMANCE METRICS:")
    print("-" * 30)
    print(f" Accuracy:           {maltese_metrics.get('accuracy', 0):.1%}")
    print(f" Precision:          {maltese_metrics.get('precision', 0):.1%}")
    print(f" Recall:             {maltese_metrics.get('recall', 0):.1%}")
    print(f" F1-Score:           {maltese_metrics.get('f1_score', 0):.3f}")
    print(f" Samples evaluated: {int(maltese_metrics.get('samples_evaluated', 0))}")
    print(f" Support:            {int(maltese_metrics.get('support', 0))}")
    
    print("\n CONFIDENCE ANALYSIS:")
    print("-" * 30)
    avg_conf = maltese_metrics.get('avg_confidence', 0)
    std_conf = maltese_metrics.get('std_confidence', 0)
    min_conf = maltese_complete.get('min_confidence', 0)
    max_conf = maltese_complete.get('max_confidence', 0)
    
    print(f" Average confidence: {avg_conf:.1%}")
    print(f" Std. deviation:     {std_conf:.4f}")
    print(f" Min confidence:     {min_conf:.1%}")
    print(f" Max confidence:     {max_conf:.1%}")
    
    # Calculate percentiles compared to all breeds
    all_f1_scores = [metrics.get('f1_score', 0) for metrics in class_metrics.values()]
    all_accuracies = [metrics.get('accuracy', 0) for metrics in class_metrics.values()]
    all_confidences = [metrics.get('avg_confidence', 0) for metrics in class_metrics.values()]
    
    maltese_f1 = maltese_metrics.get('f1_score', 0)
    maltese_accuracy = maltese_metrics.get('accuracy', 0)
    maltese_confidence = maltese_metrics.get('avg_confidence', 0)
    
    # Percentiles
    f1_percentile = (sum(1 for score in all_f1_scores if score < maltese_f1) / len(all_f1_scores)) * 100
    acc_percentile = (sum(1 for acc in all_accuracies if acc < maltese_accuracy) / len(all_accuracies)) * 100
    conf_percentile = (sum(1 for conf in all_confidences if conf < maltese_confidence) / len(all_confidences)) * 100
    
    print("\n COMPARISON WITH OTHER BREEDS:")
    print("-" * 35)
    print(f" F1-Score:    Top {100-f1_percentile:.0f}% (Percentile {f1_percentile:.0f})")
    print(f" Accuracy:    Top {100-acc_percentile:.0f}% (Percentile {acc_percentile:.0f})")
    print(f" Confidence:   Top {100-conf_percentile:.0f}% (Percentile {conf_percentile:.0f})")
    
    # Specific bias analysis
    print("\n SPECIFIC BIAS ANALYSIS:")
    print("-" * 35)
    
    # Identify reasons for good performance
    reasons = []
    if maltese_f1 > 0.90:
        reasons.append(" Excellent F1-Score (>0.90)")
    if maltese_accuracy == 1.0:
        reasons.append(" Perfect accuracy (100%)")
    if std_conf < 0.01:
        reasons.append(" Very low confidence variance")
    if avg_conf > 0.99:
        reasons.append(" Very high average confidence (>99%)")
    
    potential_issues = []
    if maltese_metrics.get('precision', 0) < maltese_metrics.get('recall', 0):
        potential_issues.append(" Precision lower than Recall (possible false positives)")
    if std_conf > 0.1:
        potential_issues.append(" High confidence variance")
    
    print("\n IDENTIFIED STRENGTHS:")
    for reason in reasons:
        print(f"  {reason}")
    
    if potential_issues:
        print("\n POTENTIAL AREAS FOR IMPROVEMENT:")
        for issue in potential_issues:
            print(f"  {issue}")
    else:
        print("\n NO SIGNIFICANT ISSUES IDENTIFIED")
    
    # Comparison with similar small dogs
    small_dogs = ['toy_terrier', 'papillon', 'Japanese_spaniel', 'Pomeranian', 'Chihuahua']
    available_small_dogs = {name: metrics for name, metrics in class_metrics.items() 
                           if any(small in name.lower() for small in ['toy', 'papillon', 'japanese', 'pomeranian', 'chihuahua'])}
    
    print(f"\n COMPARISON WITH SIMILAR SMALL DOGS:")
    print("-" * 45)
    print(f"{'Breed':25} | {'F1':6} | {'Acc':6} | {'Conf':6}")
    print("-" * 45)
    print(f"{'Maltese_dog':25} | {maltese_f1:.3f} | {maltese_accuracy:.3f} | {maltese_confidence:.3f}")
    
    for breed, metrics in available_small_dogs.items():
        f1 = metrics.get('f1_score', 0)
        acc = metrics.get('accuracy', 0)
        conf = metrics.get('avg_confidence', 0)
        print(f"{breed[:25]:25} | {f1:.3f} | {acc:.3f} | {conf:.3f}")
    
    # Conclusions
    print("\n" + "="*50)
    print(" MALTESE DOG CONCLUSIONS")
    print("="*50)
    
    if maltese_f1 > 0.90 and maltese_accuracy > 0.95:
        print(" EXCELLENT PERFORMANCE - Maltese shows one of the best results")
        print(" Virtually no bias detected")
        print(" Model is highly reliable for this breed")
        
        if std_conf < 0.01:
            print(" Very consistent and reliable predictions")
            
        if maltese_accuracy == 1.0:
            print(" PERFECT: 100% accuracy on test set")
    
    elif maltese_f1 > 0.80:
        print(" GOOD PERFORMANCE - Maltese has solid results")
        print(" Minimal bias detected")
    
    else:
        print(" SUBOPTIMAL PERFORMANCE - Possible bias present")
    
    print(f"\n RECOMMENDATION: {'MAINTAIN current model' if maltese_f1 > 0.90 else 'Consider improvements'}")
    
    return {
        'breed': 'Maltese_dog',
        'performance_level': 'EXCELENTE' if maltese_f1 > 0.90 else 'BUENO' if maltese_f1 > 0.80 else 'REGULAR',
        'bias_level': 'NINGUNO' if maltese_f1 > 0.90 and std_conf < 0.01 else 'BAJO',
        'metrics': maltese_metrics,
        'percentiles': {
            'f1': f1_percentile,
            'accuracy': acc_percentile,
            'confidence': conf_percentile
        }
    }

if __name__ == "__main__":
    result = analyze_maltese_performance()