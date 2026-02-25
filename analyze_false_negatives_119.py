#!/usr/bin/env python3
"""
False Negative Analysis Module for 119-Class Dog Breed Classifier
==================================================================

This module analyzes false negatives in the 119-class dog breed classification
model. False negatives occur when the model fails to correctly identify a breed,
which can significantly impact user experience.

Key Analysis Features:
- Identifies breeds with low recall (high false negative rate)
- Categorizes probable causes of false negatives
- Analyzes recall vs precision balance per breed
- Generates actionable recommendations for improvement

The analysis helps prioritize which breeds need:
- Threshold adjustments
- Additional training data
- Specialized model attention

Author: Dog Breed Classifier Team
Date: 2024
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

class FalseNegativeAnalyzer:
    """
    Analyzer for identifying and categorizing false negatives in breed classification.
    
    This class loads model metrics and performs comprehensive analysis of breeds
    that generate high rates of false negatives, helping to identify patterns
    and recommend mitigation strategies.
    
    Attributes:
        class_metrics (dict): Per-class performance metrics loaded from JSON.
    """
    
    def __init__(self):
        """Initialize the analyzer and load class metrics."""
        self.class_metrics = {}
        self.load_data()
        
    def load_data(self):
        """
        Load class metrics from the JSON data file.
        
        Loads performance metrics including precision, recall, F1-score,
        and confidence statistics for each breed class.
        
        Raises:
            Exception: If class_metrics.json cannot be loaded.
        """
        try:
            with open('class_metrics.json', 'r') as f:
                self.class_metrics = json.load(f)
            
            print(f" Loaded metrics for {len(self.class_metrics)} classes")
            
        except Exception as e:
            print(f" Error loading data: {e}")
    
    def analyze_false_negatives(self):
        """
        Analyze breeds with high false negative risk (low recall).
        
        Calculates false negative rates for all breeds and identifies
        the worst performers that need immediate attention.
        
        Returns:
            tuple: (list of worst recall breeds, DataFrame with all metrics)
        """
        print("\n" + "="*70)
        print(" FALSE NEGATIVES ANALYSIS - LOW RECALL")
        print("="*70)
        
        # Implementation note.
        df_data = []
        for breed, metrics in self.class_metrics.items():
            recall = metrics.get('recall', 0)
            precision = metrics.get('precision', 0)
            f1_score = metrics.get('f1_score', 0)
            support = metrics.get('support', 0)
            accuracy = metrics.get('accuracy', 0)
            avg_confidence = metrics.get('avg_confidence', 0)
            std_confidence = metrics.get('std_confidence', 0)
            
            # Calculate approximate false negatives
            true_positives = recall * support
            false_negatives = support - true_positives
            false_negative_rate = false_negatives / support if support > 0 else 0
            
            df_data.append({
                'breed': breed,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score,
                'support': support,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'std_confidence': std_confidence,
                'false_negatives': false_negatives,
                'false_negative_rate': false_negative_rate
            })
        
        df = pd.DataFrame(df_data)
        
        # Sort by recall ascending (worst first)
        df_sorted = df.sort_values('recall')
        
        print("\n TOP 15 BREEDS WITH MOST FALSE NEGATIVES (Lowest Recall):")
        print("=" * 80)
        print(f"{'Breed':25} | {'Recall':6} | {'FN':3} | {'FN%':5} | {'Prec':6} | {'F1':6} | {'Conf':6}")
        print("=" * 80)
        
        worst_recall_breeds = []
        
        for idx, row in df_sorted.head(15).iterrows():
            breed = row['breed']
            recall = row['recall']
            fn_count = int(row['false_negatives'])
            fn_rate = row['false_negative_rate']
            precision = row['precision']
            f1 = row['f1_score']
            confidence = row['avg_confidence']
            
            # Classify severity
            if recall < 0.50:
                severity = " CRITICAL"
            elif recall < 0.70:
                severity = " HIGH"
            elif recall < 0.85:
                severity = " MEDIUM"
            else:
                severity = " LOW"
            
            print(f"{breed[:24]:25} | {recall:.3f} | {fn_count:3} | {fn_rate:.1%} | {precision:.3f} | {f1:.3f} | {confidence:.3f}")
            
            worst_recall_breeds.append({
                'breed': breed,
                'recall': recall,
                'false_negatives': fn_count,
                'severity': severity,
                'precision': precision,
                'f1_score': f1
            })
        
        return worst_recall_breeds, df
    
    def categorize_false_negative_causes(self, worst_breeds):
        """
        Categorize probable causes of false negatives by breed groups.
        
        Groups problematic breeds by visual similarity to identify
        patterns in false negative occurrences.
        
        Args:
            worst_breeds (list): List of breeds with worst recall metrics.
        
        Returns:
            dict: Group problems mapping group names to affected breeds.
        """
        print("\n" + "="*70)
        print(" FALSE NEGATIVE CAUSE ANALYSIS")
        print("="*70)
        
        # Implementation note.
        similar_groups = {
            "Small Terriers": [
                "Norfolk_terrier", "Norwich_terrier", "cairn", "Yorkshire_terrier",
                "West_Highland_white_terrier", "Scottish_terrier", "Australian_terrier",
                "toy_terrier", "Lakeland_terrier", "Border_terrier"
            ],
            "Nordic Dogs/Spitz": [
                "Siberian_husky", "malamute", "Samoyed", "Eskimo_dog",
                "Norwegian_elkhound", "Pomeranian", "keeshond", "chow"
            ],
            "Sighthounds": [
                "whippet", "Italian_greyhound", "Afghan_hound", "borzoi",
                "Ibizan_hound", "Saluki", "Scottish_deerhound"
            ],
            "Spaniels": [
                "cocker_spaniel", "English_springer", "Welsh_springer_spaniel",
                "Japanese_spaniel", "Blenheim_spaniel", "Sussex_spaniel"
            ],
            "Shepherds": [
                "German_shepherd", "collie", "Border_collie", "Shetland_sheepdog",
                "Old_English_sheepdog", "malinois", "groenendael"
            ]
        }
        
        print("\n CATEGORIZATION BY PROBLEMATIC GROUPS:")
        print("-" * 50)
        
        group_problems = {}
        
        for group_name, breeds in similar_groups.items():
            group_false_negatives = []
            
            for breed_data in worst_breeds:
                breed = breed_data['breed']
                if breed in breeds:
                    group_false_negatives.append(breed_data)
            
            if group_false_negatives:
                group_problems[group_name] = group_false_negatives
                
                print(f"\n Group: {group_name}")
                print(f"   Problematic breeds: {len(group_false_negatives)}")
                
                for breed_data in group_false_negatives:
                    breed = breed_data['breed']
                    recall = breed_data['recall']
                    fn_count = breed_data['false_negatives']
                    severity = breed_data['severity']
                    
                    print(f"   - {breed:20} | Recall: {recall:.3f} | FN: {fn_count:2} | {severity}")
        
        return group_problems
    
    def analyze_recall_vs_precision_balance(self, df):
        """
        Analyze the balance between recall and precision per breed.
        
        Identifies cases where the model is overly conservative
        (high precision, low recall) which leads to false negatives.
        
        Args:
            df (pd.DataFrame): DataFrame with breed metrics.
        
        Returns:
            pd.DataFrame: Breeds with recall << precision imbalance.
        """
        print("\n" + "="*70)
        print("  RECALL vs PRECISION BALANCE ANALYSIS")
        print("="*70)
        
        # Identify cases where recall << precision (many false negatives)
        df['recall_precision_diff'] = df['precision'] - df['recall']
        
        # Cases where precision is much higher than recall
        high_imbalance = df[df['recall_precision_diff'] > 0.2].sort_values('recall_precision_diff', ascending=False)
        
        print("\n BREEDS WITH RECALL << PRECISION IMBALANCE:")
        print("   (Model too conservative - generates many false negatives)")
        print("-" * 65)
        print(f"{'Breed':25} | {'Recall':6} | {'Prec':6} | {'Diff':6} | {'Interpretation'}")
        print("-" * 65)
        
        for idx, row in high_imbalance.head(10).iterrows():
            breed = row['breed']
            recall = row['recall']
            precision = row['precision']
            diff = row['recall_precision_diff']
            
            if diff > 0.4:
                interpretation = "VERY CONSERVATIVE"
            elif diff > 0.3:
                interpretation = "CONSERVATIVE"
            elif diff > 0.2:
                interpretation = "SOMEWHAT CONSERVATIVE"
            else:
                interpretation = "BALANCED"
            
            print(f"{breed[:24]:25} | {recall:.3f} | {precision:.3f} | {diff:+.3f} | {interpretation}")
        
        return high_imbalance
    
    def generate_false_negative_recommendations(self, worst_breeds, group_problems, imbalanced_breeds):
        """
        Generate specific recommendations to reduce false negatives.
        
        Provides actionable strategies based on the analysis results,
        including threshold adjustments, data augmentation approaches,
        and training modifications.
        
        Args:
            worst_breeds (list): Breeds with worst recall metrics.
            group_problems (dict): Problematic breed groups.
            imbalanced_breeds (pd.DataFrame): Breeds with recall/precision imbalance.
        
        Returns:
            dict: Complete recommendations report.
        """
        print("\n" + "="*70)
        print(" RECOMMENDATIONS TO REDUCE FALSE NEGATIVES")
        print("="*70)
        
        # Classify by priority
        critical_breeds = [b for b in worst_breeds if b['recall'] < 0.60]
        high_priority_breeds = [b for b in worst_breeds if 0.60 <= b['recall'] < 0.75]
        
        print(f"\n CRITICAL ATTENTION ({len(critical_breeds)} breeds with Recall < 0.60):")
        print("-" * 50)
        for breed_data in critical_breeds:
            breed = breed_data['breed']
            recall = breed_data['recall']
            fn_count = breed_data['false_negatives']
            print(f"   {breed:25} | Recall: {recall:.3f} | FN: {fn_count:2}")
        
        print(f"\n  HIGH PRIORITY ({len(high_priority_breeds)} breeds with Recall 0.60-0.75):")
        print("-" * 50)
        for breed_data in high_priority_breeds:
            breed = breed_data['breed']
            recall = breed_data['recall']
            fn_count = breed_data['false_negatives']
            print(f"   {breed:25} | Recall: {recall:.3f} | FN: {fn_count:2}")
        
        print("\n  SPECIFIC STRATEGIES:")
        print("-" * 40)
        
        strategies = [
            "1.  LOWER THRESHOLD for conservative breeds",
            "2.  WEIGHTED LOSS function with extra penalty for false negatives",
            "3.  DATA AUGMENTATION specific for breeds with few detected samples",
            "4.  FOCAL LOSS to balance difficult classes",
            "5.  ENSEMBLE METHODS to improve sensitivity",
            "6.  FEATURE ENHANCEMENT for distinctive characteristics",
            "7.   THRESHOLD TUNING per individual breed",
            "8.  HARD NEGATIVE MINING for difficult cases"
        ]
        
        for strategy in strategies:
            print(f"  {strategy}")
        
        print("\n IMMEDIATE ACTIONS BY GROUP:")
        print("-" * 35)
        
        if "Small Terriers" in group_problems:
            print("   SMALL TERRIERS:")
            print("     - Focus on subtle ear and coat differences")
            print("     - Augmentation with angle and posture variations")
        
        if "Nordic Dogs/Spitz" in group_problems:
            print("    NORDIC DOGS:")
            print("     - Highlight size and tail shape differences")
            print("     - More data from different seasons/backgrounds")
        
        if "Sighthounds" in group_problems:
            print("   SIGHTHOUNDS:")
            print("     - Focus on specific body proportions")
            print("     - Full body images, not just head")
        
        # Save reporte
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'analysis_type': 'false_negatives',
            'total_breeds_analyzed': len(self.class_metrics),
            'critical_breeds': [b['breed'] for b in critical_breeds],
            'high_priority_breeds': [b['breed'] for b in high_priority_breeds],
            'worst_recall_breeds': worst_breeds,
            'group_problems': {k: [b['breed'] for b in v] for k, v in group_problems.items()},
            'recommendations': strategies
        }
        
        with open('false_negatives_analysis_119.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n Report saved to: false_negatives_analysis_119.json")
        
        return report_data

def main():
    """
    Main entry point for false negative analysis.
    
    Initializes the analyzer and runs the complete false negative
    analysis pipeline, generating recommendations report.
    """
    print(" Starting False Negative Analysis - 119 Class Model...")
    
    analyzer = FalseNegativeAnalyzer()
    
    if not analyzer.class_metrics:
        print(" Could not load metrics. Verify that class_metrics.json exists.")
        return
    
    # Run complete analysis pipeline
    worst_breeds, df = analyzer.analyze_false_negatives()
    
    # Categorize causes
    group_problems = analyzer.categorize_false_negative_causes(worst_breeds)
    
    # Analyze recall vs precision balance
    imbalanced_breeds = analyzer.analyze_recall_vs_precision_balance(df)
    
    # Generate recommendations
    report = analyzer.generate_false_negative_recommendations(worst_breeds, group_problems, imbalanced_breeds)
    
    print("\n False negative analysis completed!")
    print(f" {len(worst_breeds)} breeds identified with recall issues")
    print(f" {len([b for b in worst_breeds if b['recall'] < 0.60])} breeds need critical attention")

if __name__ == "__main__":
    main()