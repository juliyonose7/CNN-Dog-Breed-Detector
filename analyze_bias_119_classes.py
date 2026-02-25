#!/usr/bin/env python3
"""
Bias Analysis Module for 119-Class Dog Breed Classifier
========================================================

This module performs comprehensive bias analysis on a 119-class dog breed
classification model. It identifies breeds with the highest probability of
bias based on:

- Performance metrics per class (precision, recall, F1-score)
- Visual similarities between breeds that may cause misclassification
- Geographic/regional representation bias in the dataset
- Training data distribution patterns

The analysis helps identify problematic breeds that may need:
- Additional training data
- Data augmentation strategies
- Specialized model attention
- Threshold adjustments

Author: Dog Breed Classifier Team
Date: 2024
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

class BiasAnalyzer119:
    """
    Comprehensive bias analyzer for 119-class dog breed classification model.
    
    This class analyzes various types of bias in the model's predictions,
    including performance-based bias, visual similarity bias, and geographic
    representation bias.
    
    Attributes:
        class_metrics (dict): Per-class performance metrics loaded from JSON.
        breed_names (list): List of breed names from the balanced model.
    """
    
    def __init__(self):
        """Initialize the bias analyzer and load required data."""
        self.class_metrics = {}
        self.breed_names = []
        self.load_data()
        
    def load_data(self):
        """
        Load class metrics and breed names from data files.
        
        Loads performance metrics from 'class_metrics.json' and retrieves
        breed names from the balanced model server configuration.
        
        Raises:
            Exception: If data files cannot be loaded.
        """
        try:
            # Implementation note.
            with open('class_metrics.json', 'r') as f:
                self.class_metrics = json.load(f)
            
            # Get names of breeds of the balanced model
            from balanced_model_server import CLASS_NAMES
            self.breed_names = [name.split('-')[1] if '-' in name else name for name in CLASS_NAMES]
            
            print(f"✅ Loaded metrics for {len(self.class_metrics)} classes")
            print(f"✅ Retrieved {len(self.breed_names)} breed names from model")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def analyze_performance_bias(self):
        """
        Analyze bias based on per-class performance metrics.
        
        Identifies breeds with poor performance (low F1-score, recall, etc.)
        and high confidence variance, which may indicate model bias.
        
        Returns:
            pd.DataFrame: DataFrame containing performance metrics for all breeds,
                         sorted and analyzed for bias indicators.
        """
        print("\n" + "="*60)
        print("📊 PERFORMANCE-BASED BIAS ANALYSIS")
        print("="*60)
        
        # Implementation note.
        df_data = []
        for breed, metrics in self.class_metrics.items():
            df_data.append({
                'breed': breed,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'accuracy': metrics.get('accuracy', 0),
                'avg_confidence': metrics.get('avg_confidence', 0),
                'std_confidence': metrics.get('std_confidence', 0),
                'support': metrics.get('support', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Identify breeds with worst performance
        print("\n🔴 BREEDS WITH HIGHEST BIAS (Worst Performance):")
        print("-" * 50)
        
        # Top 10 worst F1-Score
        worst_f1 = df.nsmallest(10, 'f1_score')
        print("\n📉 Top 10 Worst F1-Score:")
        for idx, row in worst_f1.iterrows():
            print(f"  {row['breed']:25} | F1: {row['f1_score']:.3f} | Acc: {row['accuracy']:.3f}")
        
        # Implementation note.
        worst_recall = df.nsmallest(10, 'recall')
        print("\n⚠️  Top 10 Worst Recall (Most False Negatives):")
        for idx, row in worst_recall.iterrows():
            print(f"  {row['breed']:25} | Recall: {row['recall']:.3f} | Precision: {row['precision']:.3f}")
        
        # Breeds with high confidence variance
        high_variance = df.nlargest(10, 'std_confidence')
        print("\n🌀 Top 10 Highest Confidence Variance:")
        for idx, row in high_variance.iterrows():
            print(f"  {row['breed']:25} | Std: {row['std_confidence']:.3f} | Avg: {row['avg_confidence']:.3f}")
        
        return df
    
    def analyze_visual_similarity_bias(self):
        """
        Analyze bias caused by visual similarities between breed groups.
        
        Groups visually similar breeds and analyzes if the model shows
        higher confusion rates within these groups, indicating potential
        visual similarity bias.
        
        Returns:
            dict: Dictionary containing bias risk assessment for each
                 visually similar breed group.
        """
        print("\n" + "="*60)
        print("👁️  VISUAL SIMILARITY BIAS ANALYSIS")
        print("="*60)
        
        # Implementation note.
        similar_groups = {
            "Small Terriers": [
                "Yorkshire_terrier", "cairn", "Norfolk_terrier", "Norwich_terrier",
                "West_Highland_white_terrier", "Scottish_terrier", "Australian_terrier"
            ],
            "Spaniels": [
                "Japanese_spaniel", "Blenheim_spaniel", "cocker_spaniel", 
                "English_springer", "Welsh_springer_spaniel", "Sussex_spaniel"
            ],
            "Shepherds/Collies": [
                "collie", "Border_collie", "Shetland_sheepdog", "Old_English_sheepdog",
                "German_shepherd", "malinois", "groenendael"
            ],
            "Nordic Dogs": [
                "Siberian_husky", "malamute", "Samoyed", "Eskimo_dog",
                "Norwegian_elkhound", "Pomeranian"
            ],
            "Sighthounds": [
                "Afghan_hound", "borzoi", "Italian_greyhound", "Ibizan_hound",
                "Saluki", "Scottish_deerhound", "whippet"
            ],
            "Bulldogs/Mastiffs": [
                "French_bulldog", "Boston_bull", "bull_mastiff", "Great_Dane",
                "Saint_Bernard", "Tibetan_mastiff"
            ],
            "Retrievers": [
                "golden_retriever", "Labrador_retriever", "flat-coated_retriever",
                "curly-coated_retriever", "Chesapeake_Bay_retriever"
            ],
            "Schnauzers": [
                "miniature_schnauzer", "giant_schnauzer", "standard_schnauzer"
            ]
        }
        
        bias_risk = {}
        
        for group_name, breeds in similar_groups.items():
            print(f"\n🔍 Grupo: {group_name}")
            group_metrics = []
            available_breeds = []
            
            for breed in breeds:
                if breed in self.class_metrics:
                    metrics = self.class_metrics[breed]
                    group_metrics.append({
                        'breed': breed,
                        'f1_score': metrics.get('f1_score', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0)
                    })
                    available_breeds.append(breed)
            
            if group_metrics:
                # Calculate group variance
                f1_scores = [m['f1_score'] for m in group_metrics]
                f1_variance = np.var(f1_scores)
                f1_mean = np.mean(f1_scores)
                
                bias_risk[group_name] = {
                    'variance': f1_variance,
                    'mean_f1': f1_mean,
                    'breeds': available_breeds,
                    'risk_level': 'HIGH' if f1_variance > 0.05 else 'MEDIUM' if f1_variance > 0.02 else 'LOW'
                }
                
                print(f"  📊 Average F1: {f1_mean:.3f}")
                print(f"  🌀 F1 Variance: {f1_variance:.4f}")
                print(f"  ⚠️  Bias Risk: {bias_risk[group_name]['risk_level']}")
                
                # Show worst performing in group
                worst_in_group = sorted(group_metrics, key=lambda x: x['f1_score'])[:3]
                print(f"  🔴 Worst in group:")
                for breed_data in worst_in_group:
                    print(f"    - {breed_data['breed']:20} F1: {breed_data['f1_score']:.3f}")
        
        return bias_risk
    
    def analyze_geographic_bias(self):
        """
        Analyze geographic/regional bias in the model's performance.
        
        Groups breeds by their geographic origin and analyzes if certain
        regions are over or under-represented in terms of model performance.
        
        Returns:
            dict: Regional performance statistics including mean F1-score,
                 standard deviation, and breed counts per region.
        """
        print("\n" + "="*60)
        print("🌍 GEOGRAPHIC BIAS ANALYSIS")
        print("="*60)
        
        # Implementation note.
        geographic_regions = {
            "Western Europe": [
                "German_shepherd", "Rottweiler", "Doberman", "Great_Dane", "boxer",
                "German_short-haired_pointer", "Weimaraner", "giant_schnauzer",
                "standard_schnauzer", "miniature_schnauzer", "Bernese_mountain_dog"
            ],
            "United Kingdom": [
                "English_foxhound", "English_setter", "English_springer", "cocker_spaniel",
                "Yorkshire_terrier", "West_Highland_white_terrier", "Scottish_terrier",
                "Border_collie", "collie", "Shetland_sheepdog", "cairn", "Norfolk_terrier",
                "Norwich_terrier", "Airedale", "Border_terrier", "Bedlington_terrier"
            ],
            "France": [
                "Brittany_spaniel", "papillon", "Bouvier_des_Flandres", "briard",
                "French_bulldog"
            ],
            "Scandinavia": [
                "Norwegian_elkhound", "Siberian_husky", "malamute", "Samoyed",
                "Eskimo_dog"
            ],
            "Asia": [
                "chow", "Pomeranian", "Japanese_spaniel", "Shih-Tzu", "Lhasa",
                "Tibetan_terrier", "Tibetan_mastiff", "basenji"  # basenji is African
            ],
            "Mediterranean": [
                "Italian_greyhound", "Ibizan_hound", "Saluki"
            ],
            "Americas": [
                "American_Staffordshire_terrier", "Boston_bull", "Chesapeake_Bay_retriever"
            ]
        }
        
        regional_performance = {}
        
        for region, breeds in geographic_regions.items():
            f1_scores = []
            available_breeds = []
            
            for breed in breeds:
                if breed in self.class_metrics:
                    f1_scores.append(self.class_metrics[breed].get('f1_score', 0))
                    available_breeds.append(breed)
            
            if f1_scores:
                regional_performance[region] = {
                    'mean_f1': np.mean(f1_scores),
                    'std_f1': np.std(f1_scores),
                    'count': len(f1_scores),
                    'breeds': available_breeds
                }
        
        print("\n📊 Performance by Region:")
        print("-" * 40)
        sorted_regions = sorted(regional_performance.items(), 
                              key=lambda x: x[1]['mean_f1'], reverse=True)
        
        for region, data in sorted_regions:
            print(f"{region:20} | F1: {data['mean_f1']:.3f} ± {data['std_f1']:.3f} | Breeds: {data['count']}")
        
        # Identify regions with bias
        all_f1_means = [data['mean_f1'] for data in regional_performance.values()]
        global_mean = np.mean(all_f1_means)
        
        print(f"\n🎯 Global Average F1: {global_mean:.3f}")
        print("\n⚠️  Regions with Potential Bias:")
        print("-" * 40)
        
        for region, data in sorted_regions:
            if data['mean_f1'] < global_mean - 0.05:
                print(f"🔴 {region}: {data['mean_f1']:.3f} (UNDERPERFORMING)")
            elif data['mean_f1'] > global_mean + 0.05:
                print(f"🟢 {region}: {data['mean_f1']:.3f} (OVERREPRESENTED)")
        
        return regional_performance
    
    def generate_bias_report(self):
        """
        Generate comprehensive bias analysis report.
        
        Runs all bias analysis methods and compiles results into a
        complete report, identifying high-risk breeds and providing
        actionable recommendations.
        
        Returns:
            dict: Complete bias analysis report containing performance
                 metrics, risk assessments, and recommendations.
        """
        print("\n" + "="*70)
        print("📋 COMPLETE BIAS ANALYSIS REPORT - 119 CLASSES")
        print("="*70)
        
        # Implementation note.
        df_performance = self.analyze_performance_bias()
        visual_bias = self.analyze_visual_similarity_bias()
        geographic_bias = self.analyze_geographic_bias()
        
        # Implementation note.
        print("\n" + "="*60)
        print("🎯 BREEDS WITH HIGHEST BIAS RISK")
        print("="*60)
        
        # Implementation note.
        high_risk_breeds = set()
        
        # From performance (worst 15)
        worst_performers = df_performance.nsmallest(15, 'f1_score')['breed'].tolist()
        high_risk_breeds.update(worst_performers)
        
        # From visual similarity (high risk groups)
        for group, data in visual_bias.items():
            if data['risk_level'] == 'HIGH':
                high_risk_breeds.update(data['breeds'])
        
        print(f"\n🚨 TOP HIGH-RISK BREEDS ({len(high_risk_breeds)} total):")
        print("-" * 50)
        
        for i, breed in enumerate(sorted(high_risk_breeds), 1):
            if breed in self.class_metrics:
                metrics = self.class_metrics[breed]
                f1 = metrics.get('f1_score', 0)
                recall = metrics.get('recall', 0)
                precision = metrics.get('precision', 0)
                print(f"{i:2}. {breed:25} | F1: {f1:.3f} | P: {precision:.3f} | R: {recall:.3f}")
        
        # Implementation note.
        print("\n" + "="*60)
        print("💡 SPECIFIC RECOMMENDATIONS")
        print("="*60)
        
        recommendations = [
            "1. 🎯 PRIORITY FOCUS on breeds with F1 < 0.70",
            "2. 🔄 INCREASE training data for low-performing breeds",
            "3. 👁️  DIFFERENTIATION TECHNIQUES for visually similar groups",
            "4. 🌍 BALANCE geographic representation in the dataset",
            "5. 🧠 SPECIFIC FINE-TUNING for problematic breeds",
            "6. 📊 ADAPTIVE THRESHOLDS per breed based on historical performance",
            "7. 🔍 SPECIALIZED AUGMENTATION for confused breeds",
            "8. ⚖️  WEIGHTED LOSS per class during retraining"
        ]
        
        for rec in recommendations:
            print(rec)
        
        # Save reporte
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_classes': len(self.class_metrics),
            'high_risk_breeds': list(high_risk_breeds),
            'performance_metrics': df_performance.to_dict('records'),
            'visual_similarity_risk': visual_bias,
            'geographic_performance': geographic_bias,
            'recommendations': recommendations
        }
        
        with open('bias_analysis_119_classes.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Report saved to: bias_analysis_119_classes.json")
        
        return report_data

def main():
    """
    Main entry point for bias analysis.
    
    Initializes the BiasAnalyzer119 and generates a comprehensive
    bias report for the 119-class model.
    """
    print("🔍 Starting Bias Analysis for 119-Class Model...")
    
    analyzer = BiasAnalyzer119()
    
    if not analyzer.class_metrics:
        print("❌ Could not load metrics. Verify that class_metrics.json exists.")
        return
    
    # Generate complete report
    report = analyzer.generate_bias_report()
    
    print("\n✅ Bias analysis completed successfully!")

if __name__ == "__main__":
    main()