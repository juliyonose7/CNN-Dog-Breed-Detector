#!/usr/bin/env python3
"""
Comprehensive Bias Detection and Analysis Module
=================================================

This module analyzes different types of biases in the dog breed
classification system across multiple dimensions:

1. Dataset Representation Bias - Class imbalance analysis
2. Geographical/Cultural Bias - Regional distribution of breeds
3. Cultural Popularity Bias - Representation by breed popularity
4. Physical Characteristics Bias - Size, coat, color distributions  
5. Model Architecture Bias - Different model advantages
6. Evaluation Bias - Metric and threshold fairness

The analysis generates:
- Visual reports as PNG files
- JSON reports with detailed findings
- Mitigation strategy recommendations

Author: Dog Breed Classifier Team
Date: 2024
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class BiasDetectionAnalyzer:
    """
    Comprehensive bias detection analyzer for dog breed classification.
    
    Analyzes multiple dimensions of potential bias in the model and dataset,
    including geographical, physical, architectural, and evaluation biases.
    
    Attributes:
        workspace_path (Path): Root path of the project workspace.
        breed_data_path (Path): Path to balanced breed training data.
        yesdog_path (Path): Path to original YESDOG dataset.
        breed_geography (dict): Mapping of breeds to geographic origins.
        breed_characteristics (dict): Mapping of breeds to physical traits.
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize the bias detection analyzer.
        
        Args:
            workspace_path (str): Path to the project workspace root.
        """
        self.workspace_path = Path(workspace_path)
        self.breed_data_path = self.workspace_path / "breed_processed_data" / "train"
        self.yesdog_path = self.workspace_path / "DATASETS" / "YESDOG"
        
        # Geographic origin mapping for breeds
        self.breed_geography = {
            # Breeds Europeas
            'German_shepherd': {'region': 'Europe', 'country': 'Germany', 'popularity': 'very_high'},
            'Rottweiler': {'region': 'Europe', 'country': 'Germany', 'popularity': 'high'},
            'Doberman': {'region': 'Europe', 'country': 'Germany', 'popularity': 'high'},
            'Great_Dane': {'region': 'Europe', 'country': 'Germany', 'popularity': 'medium'},
            'Weimaraner': {'region': 'Europe', 'country': 'Germany', 'popularity': 'medium'},
            'Scottish_deerhound': {'region': 'Europe', 'country': 'Scotland', 'popularity': 'low'},
            'Border_collie': {'region': 'Europe', 'country': 'UK', 'popularity': 'very_high'},
            'English_setter': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Irish_setter': {'region': 'Europe', 'country': 'Ireland', 'popularity': 'medium'},
            'Airedale': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Norfolk_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Norwich_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Yorkshire_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'very_high'},
            'Bedlington_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Border_terrier': {'region': 'Europe', 'country': 'UK', 'popularity': 'medium'},
            'Kerry_blue_terrier': {'region': 'Europe', 'country': 'Ireland', 'popularity': 'low'},
            'Lakeland_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Sealyham_terrier': {'region': 'Europe', 'country': 'Wales', 'popularity': 'low'},
            'Saint_Bernard': {'region': 'Europe', 'country': 'Switzerland', 'popularity': 'high'},
            'Bernese_mountain_dog': {'region': 'Europe', 'country': 'Switzerland', 'popularity': 'high'},
            'EntleBucher': {'region': 'Europe', 'country': 'Switzerland', 'popularity': 'low'},
            'Great_Pyrenees': {'region': 'Europe', 'country': 'France', 'popularity': 'medium'},
            'Newfoundland': {'region': 'North America', 'country': 'Canada', 'popularity': 'medium'},
            'Italian_greyhound': {'region': 'Europe', 'country': 'Italy', 'popularity': 'medium'},
            'Norwegian_elkhound': {'region': 'Europe', 'country': 'Norway', 'popularity': 'medium'},
            
            # Asian breeds
            'Shih-Tzu': {'region': 'Asia', 'country': 'China', 'popularity': 'very_high'},
            'Lhasa': {'region': 'Asia', 'country': 'Tibet', 'popularity': 'medium'},
            'Pug': {'region': 'Asia', 'country': 'China', 'popularity': 'very_high'},
            'Japanese_spaniel': {'region': 'Asia', 'country': 'Japan', 'popularity': 'medium'},
            'Tibetan_terrier': {'region': 'Asia', 'country': 'Tibet', 'popularity': 'low'},
            'chow': {'region': 'Asia', 'country': 'China', 'popularity': 'medium'},
            'Samoyed': {'region': 'Asia', 'country': 'Russia', 'popularity': 'high'},
            'Siberian_husky': {'region': 'Asia', 'country': 'Russia', 'popularity': 'very_high'},
            
            # Breeds Americanas
            'Boston_bull': {'region': 'North America', 'country': 'USA', 'popularity': 'high'},
            'Labrador_retriever': {'region': 'North America', 'country': 'Canada', 'popularity': 'very_high'},
            'Chesapeake_Bay_retriever': {'region': 'North America', 'country': 'USA', 'popularity': 'medium'},
            
            # Middle Eastern and African breeds
            'Afghan_hound': {'region': 'Middle East', 'country': 'Afghanistan', 'popularity': 'medium'},
            'Saluki': {'region': 'Middle East', 'country': 'Middle East', 'popularity': 'low'},
            'basenji': {'region': 'Africa', 'country': 'Central Africa', 'popularity': 'low'},
            
            # Breeds Australianas
            'Australian_terrier': {'region': 'Oceania', 'country': 'Australia', 'popularity': 'medium'},
            
            # Additional common breeds
            'beagle': {'region': 'Europe', 'country': 'England', 'popularity': 'very_high'},
            'basset': {'region': 'Europe', 'country': 'France', 'popularity': 'high'},
            'bloodhound': {'region': 'Europe', 'country': 'Belgium', 'popularity': 'medium'},
            'bluetick': {'region': 'North America', 'country': 'USA', 'popularity': 'low'},
            'whippet': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Ibizan_hound': {'region': 'Europe', 'country': 'Spain', 'popularity': 'low'},
            'Irish_wolfhound': {'region': 'Europe', 'country': 'Ireland', 'popularity': 'medium'},
            'Rhodesian_ridgeback': {'region': 'Africa', 'country': 'Zimbabwe', 'popularity': 'medium'},
            'Maltese_dog': {'region': 'Europe', 'country': 'Malta', 'popularity': 'high'},
            'papillon': {'region': 'Europe', 'country': 'France', 'popularity': 'medium'},
            'Pomeranian': {'region': 'Europe', 'country': 'Germany', 'popularity': 'high'},
            'silky_terrier': {'region': 'Oceania', 'country': 'Australia', 'popularity': 'low'},
            'toy_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Blenheim_spaniel': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'cairn': {'region': 'Europe', 'country': 'Scotland', 'popularity': 'medium'},
            'Dandie_Dinmont': {'region': 'Europe', 'country': 'Scotland', 'popularity': 'low'},
            'malamute': {'region': 'North America', 'country': 'USA', 'popularity': 'medium'},
            'miniature_pinscher': {'region': 'Europe', 'country': 'Germany', 'popularity': 'medium'},
            'Pembroke': {'region': 'Europe', 'country': 'Wales', 'popularity': 'high'},
            'Leonberg': {'region': 'Europe', 'country': 'Germany', 'popularity': 'low'}
        }
        
        # Physical characteristics mapping by category
        self.breed_characteristics = {
            'size': {
                'toy': ['Japanese_spaniel', 'Maltese_dog', 'papillon', 'Pomeranian', 'silky_terrier', 'toy_terrier', 'Yorkshire_terrier'],
                'small': ['beagle', 'basset', 'cairn', 'Dandie_Dinmont', 'miniature_pinscher', 'Norfolk_terrier', 'Norwich_terrier', 'Pug', 'Shih-Tzu'],
                'medium': ['Australian_terrier', 'Bedlington_terrier', 'Border_terrier', 'Boston_bull', 'chow', 'Pembroke', 'whippet'],
                'large': ['Afghan_hound', 'Airedale', 'bloodhound', 'German_shepherd', 'Labrador_retriever', 'Rottweiler', 'Samoyed'],
                'giant': ['Great_Dane', 'Great_Pyrenees', 'Irish_wolfhound', 'Newfoundland', 'Saint_Bernard']
            },
            'coat_type': {
                'short': ['beagle', 'basset', 'bloodhound', 'Boston_bull', 'German_shepherd', 'Labrador_retriever', 'Pug', 'Rottweiler', 'whippet'],
                'medium': ['Australian_terrier', 'cairn', 'chow', 'Norwegian_elkhound', 'Pembroke', 'Siberian_husky'],
                'long': ['Afghan_hound', 'Bernese_mountain_dog', 'Irish_setter', 'Maltese_dog', 'Newfoundland', 'Shih-Tzu', 'Yorkshire_terrier'],
                'curly': ['Bedlington_terrier', 'Dandie_Dinmont', 'Kerry_blue_terrier', 'Pomeranian'],
                'wire': ['Airedale', 'Border_terrier', 'Lakeland_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Sealyham_terrier']
            },
            'color_patterns': {
                'solid': ['German_shepherd', 'Rottweiler', 'chow', 'Pug', 'Samoyed'],
                'bicolor': ['Bernese_mountain_dog', 'Boston_bull', 'Border_collie', 'Saint_Bernard'],
                'tricolor': ['beagle', 'basset', 'Airedale', 'Yorkshire_terrier'],
                'spotted': ['bluetick', 'English_setter', 'German_short-haired_pointer'],
                'brindle': ['Boston_bull', 'cairn', 'whippet']
            }
        }
        
    def analyze_dataset_representation_bias(self):
        """
        Analyze representation bias in the dataset class distribution.
        
        Examines the balance of images across breed classes and calculates
        the coefficient of variation to assess dataset balance quality.
        
        Returns:
            dict: Statistics including breed counts, CV, and balance status.
        """
        print(" REPRESENTATION BIAS ANALYSIS")
        print("="*60)
        
        if not self.breed_data_path.exists():
            print(" Balanced breed dataset not found")
            return None
            
        breed_stats = {}
        total_images = 0
        
        # Count images per breed
        for breed_dir in self.breed_data_path.iterdir():
            if breed_dir.is_dir():
                images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.png"))
                count = len(images)
                breed_stats[breed_dir.name] = count
                total_images += count
        
        if not breed_stats:
            print(" No breed data found")
            return None
            
        # Calculate distribution statistics
        counts = list(breed_stats.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        cv = std_count / mean_count  # Coefficient of variation
        
        print(f" Dataset Statistics:")
        print(f"   Total breeds: {len(breed_stats)}")
        print(f"   Total images: {total_images:,}")
        print(f"   Average per breed: {mean_count:.1f}")
        print(f"   Standard deviation: {std_count:.1f}")
        print(f"   Coefficient of variation: {cv:.3f}")
        
        # Evaluate balance quality
        if cv < 0.05:
            balance_status = " PERFECTLY BALANCED"
        elif cv < 0.1:
            balance_status = " VERY WELL BALANCED"
        elif cv < 0.2:
            balance_status = " MODERATELY BALANCED"
        else:
            balance_status = " IMBALANCED"
            
        print(f"   Balance status: {balance_status}")
        
        return {
            'breed_stats': breed_stats,
            'total_images': total_images,
            'mean_count': mean_count,
            'std_count': std_count,
            'cv': cv,
            'balance_status': balance_status
        }
    
    def analyze_geographical_bias(self):
        """
        Analyze geographical and cultural bias in breed representation.
        
        Examines the distribution of breeds by region of origin and
        cultural popularity level to identify potential biases.
        
        Returns:
            dict: Regional and popularity distributions with detected biases.
        """
        print("\n GEOGRAPHICAL AND CULTURAL BIAS ANALYSIS")
        print("="*60)
        
        # Get breeds from dataset
        dataset_breeds = []
        if self.breed_data_path.exists():
            dataset_breeds = [d.name for d in self.breed_data_path.iterdir() if d.is_dir()]
        
        if not dataset_breeds:
            print(" No breeds found in dataset")
            return None
            
        # Classify breeds by region and popularity
        regional_distribution = defaultdict(list)
        popularity_distribution = defaultdict(list)
        missing_breeds = []
        
        for breed in dataset_breeds:
            if breed in self.breed_geography:
                info = self.breed_geography[breed]
                regional_distribution[info['region']].append(breed)
                popularity_distribution[info['popularity']].append(breed)
            else:
                missing_breeds.append(breed)
        
        print(f" Distribution by Region:")
        total_classified = sum(len(breeds) for breeds in regional_distribution.values())
        
        for region, breeds in regional_distribution.items():
            percentage = len(breeds) / total_classified * 100
            print(f"   {region:15}: {len(breeds):2d} breeds ({percentage:5.1f}%)")
            
        print(f"   Unclassified:      {len(missing_breeds):2d} breeds")
        
        print(f"\n Distribution by Popularity:")
        for popularity, breeds in popularity_distribution.items():
            percentage = len(breeds) / total_classified * 100
            print(f"   {popularity:12}: {len(breeds):2d} breeds ({percentage:5.1f}%)")
        
        # Detect biases
        biases_detected = []
        
        # Regional bias
        europe_pct = len(regional_distribution.get('Europe', [])) / total_classified * 100
        if europe_pct > 60:
            biases_detected.append(f" EUROPEAN BIAS: {europe_pct:.1f}% of breeds are European")
            
        asia_pct = len(regional_distribution.get('Asia', [])) / total_classified * 100
        if asia_pct < 15:
            biases_detected.append(f" ASIAN UNDERREPRESENTATION: Only {asia_pct:.1f}% of breeds are Asian")
            
        africa_pct = len(regional_distribution.get('Africa', [])) / total_classified * 100
        if africa_pct < 5:
            biases_detected.append(f" AFRICAN UNDERREPRESENTATION: Only {africa_pct:.1f}% of breeds are African")
        
        # Popularity bias
        very_high_pct = len(popularity_distribution.get('very_high', [])) / total_classified * 100
        low_pct = len(popularity_distribution.get('low', [])) / total_classified * 100
        
        if very_high_pct > 25:
            biases_detected.append(f" POPULAR BREED BIAS: {very_high_pct:.1f}% are very popular")
            
        if low_pct < 20:
            biases_detected.append(f" RARE BREED UNDERREPRESENTATION: Only {low_pct:.1f}% are low popularity")
        
        print(f"\n DETECTED BIASES:")
        if biases_detected:
            for bias in biases_detected:
                print(f"   {bias}")
        else:
            print("    No significant geographical/cultural biases detected")
        
        return {
            'regional_distribution': dict(regional_distribution),
            'popularity_distribution': dict(popularity_distribution),
            'missing_breeds': missing_breeds,
            'biases_detected': biases_detected,
            'total_classified': total_classified
        }
    
    def analyze_physical_characteristics_bias(self):
        """
        Analyze bias in physical characteristics representation.
        
        Examines the distribution of breeds by size, coat type, and
        color patterns to identify potential biases in physical traits.
        
        Returns:
            dict: Physical characteristic distributions and detected biases.
        """
        print("\n PHYSICAL CHARACTERISTICS BIAS ANALYSIS")
        print("="*60)
        
        # Get breeds from dataset
        dataset_breeds = []
        if self.breed_data_path.exists():
            dataset_breeds = [d.name for d in self.breed_data_path.iterdir() if d.is_dir()]
        
        if not dataset_breeds:
            return None
            
        # Classify by physical characteristics
        size_distribution = defaultdict(list)
        coat_distribution = defaultdict(list)
        color_distribution = defaultdict(list)
        
        for breed in dataset_breeds:
            # Classify by size
            for size, breeds in self.breed_characteristics['size'].items():
                if breed in breeds:
                    size_distribution[size].append(breed)
                    break
            
            # Classify by coat type
            for coat_type, breeds in self.breed_characteristics['coat_type'].items():
                if breed in breeds:
                    coat_distribution[coat_type].append(breed)
                    break
                    
            # Classify by color patterns
            for color_pattern, breeds in self.breed_characteristics['color_patterns'].items():
                if breed in breeds:
                    color_distribution[color_pattern].append(breed)
                    break
        
        print(f" Distribution by Size:")
        total_size_classified = sum(len(breeds) for breeds in size_distribution.values())
        for size, breeds in size_distribution.items():
            percentage = len(breeds) / len(dataset_breeds) * 100
            print(f"   {size:8}: {len(breeds):2d} breeds ({percentage:5.1f}%)")
        
        print(f"\n Distribution by Coat Type:")
        for coat_type, breeds in coat_distribution.items():
            percentage = len(breeds) / len(dataset_breeds) * 100
            print(f"   {coat_type:8}: {len(breeds):2d} breeds ({percentage:5.1f}%)")
        
        print(f"\n Distribution by Color Patterns:")
        for color_pattern, breeds in color_distribution.items():
            percentage = len(breeds) / len(dataset_breeds) * 100
            print(f"   {color_pattern:8}: {len(breeds):2d} breeds ({percentage:5.1f}%)")
        
        # Detect physical biases
        physical_biases = []
        
        # Size bias detection
        small_breeds = len(size_distribution.get('small', [])) + len(size_distribution.get('toy', []))
        large_breeds = len(size_distribution.get('large', [])) + len(size_distribution.get('giant', []))
        
        if small_breeds > large_breeds * 1.5:
            physical_biases.append(f" SMALL DOG BIAS: {small_breeds} small vs {large_breeds} large")
        elif large_breeds > small_breeds * 1.5:
            physical_biases.append(f" LARGE DOG BIAS: {large_breeds} large vs {small_breeds} small")
        
        # Coat bias detection
        long_coat = len(coat_distribution.get('long', []))
        short_coat = len(coat_distribution.get('short', []))
        
        if long_coat > short_coat * 1.5:
            physical_biases.append(f" LONG COAT BIAS: {long_coat} long coat vs {short_coat} short coat")
        elif short_coat > long_coat * 1.5:
            physical_biases.append(f" SHORT COAT BIAS: {short_coat} short coat vs {long_coat} long coat")
        
        print(f"\n PHYSICAL BIASES DETECTED:")
        if physical_biases:
            for bias in physical_biases:
                print(f"   {bias}")
        else:
            print("    No significant physical biases detected")
        
        return {
            'size_distribution': dict(size_distribution),
            'coat_distribution': dict(coat_distribution),
            'color_distribution': dict(color_distribution),
            'physical_biases': physical_biases
        }
    
    def analyze_model_architecture_bias(self):
        """
        Analyze biases introduced by model architecture choices.
        
        Examines the hybrid model system (ResNet18/34/50) and identifies
        potential unfair advantages for breeds with specialized models.
        
        Returns:
            dict: Architectural biases and selective breed characteristics.
        """""
        print("\n MODEL ARCHITECTURE BIAS ANALYSIS")
        print("="*60)
        
        # Hybrid model system analysis
        print(" Current Hybrid System:")
        print("   1. Binary Model: ResNet18 (dog/not-dog)")
        print("   2. Main Model: ResNet50 (50 breeds)")
        print("   3. Selective Model: ResNet34 (6 problematic breeds)"))
        
        # Breeds en model selective
        selective_breeds = ['basset', 'beagle', 'Labrador_retriever', 'Norwegian_elkhound', 'pug', 'Samoyed']
        
        print(f"\n Breeds with Specialized Model:")
        for breed in selective_breeds:
            print(f"   • {breed}")
        
        # Architectural biases analysis
        architectural_biases = []
        
        # 1. Different architecture bias
        architectural_biases.append(" ARCHITECTURAL BIAS: Different architectures (ResNet18/34/50) may have different capabilities")
        
        # 2. Selective model bias
        architectural_biases.append(" SPECIALIZATION BIAS: 6 breeds have dedicated model, unfair advantage")
        
        # 3. Temperature scaling bias
        architectural_biases.append(" CALIBRATION BIAS: Temperature scaling may favor certain predictions")
        
        # Selective breed analysis
        print(f"\n Selective Breed Analysis:")
        
        selective_characteristics = {
            'regions': [],
            'sizes': [],
            'popularities': []
        }
        
        for breed in selective_breeds:
            if breed in self.breed_geography:
                info = self.breed_geography[breed]
                selective_characteristics['regions'].append(info['region'])
                selective_characteristics['popularities'].append(info['popularity'])
        
        # Verify if hay patrones en the breeds selectivas
        region_counter = Counter(selective_characteristics['regions'])
        popularity_counter = Counter(selective_characteristics['popularities'])
        
        print(f"   Regional distribution: {dict(region_counter)}")
        print(f"   Popularity distribution: {dict(popularity_counter)}")
        
        # Analyze if there are patterns in selective breeds
        most_common_region = region_counter.most_common(1)[0] if region_counter else None
        if most_common_region and most_common_region[1] >= 4:
            architectural_biases.append(f" GEOGRAPHIC BIAS IN SELECTIVE: {most_common_region[1]}/6 breeds are from {most_common_region[0]}")
        
        # Bias if selective breeds have similar popularities
        most_common_popularity = popularity_counter.most_common(1)[0] if popularity_counter else None
        if most_common_popularity and most_common_popularity[1] >= 4:
            architectural_biases.append(f" POPULARITY BIAS IN SELECTIVE: {most_common_popularity[1]}/6 breeds are {most_common_popularity[0]}")
        
        print(f"\n ARCHITECTURAL BIASES DETECTED:")
        for bias in architectural_biases:
            print(f"   {bias}")
        
        return {
            'selective_breeds': selective_breeds,
            'architectural_biases': architectural_biases,
            'selective_characteristics': selective_characteristics
        }
    
    def analyze_evaluation_bias(self):
        """
        Analyze biases in evaluation methodology.
        
        Examines current evaluation metrics, thresholds, and
        methodology for potential fairness issues.
        
        Returns:
            dict: Evaluation biases and current metric values.
        """
        print("\n EVALUATION BIAS ANALYSIS")
        print("="*60)
        
        evaluation_biases = []
        
        print(" Current Evaluation Metrics:")
        print("   • General accuracy: 88.14% (main model)")
        print("   • Selective accuracy: 95.15% (6 breeds)")
        print("   • Temperature scaling: 10.0 (calibration)")
        print("   • Confidence threshold: 0.35")
        
        # Identify common evaluation biases
        evaluation_biases.extend([
            " SINGLE METRIC BIAS: Only accuracy used, ignores per-class precision/recall",
            " TEST DATASET BIAS: Is it representative of real-world cases?",
            " CALIBRATION BIAS: Temperature scaling may mask real problems",
            " THRESHOLD BIAS: Single threshold (0.35) may not be optimal for all breeds",
            " UNFAIR COMPARISON BIAS: Selective model vs main model is not a fair comparison"
        ])
        
        print(f"\n EVALUATION BIASES DETECTED:")
        for bias in evaluation_biases:
            print(f"   {bias}")
        
        return {
            'evaluation_biases': evaluation_biases,
            'current_metrics': {
                'main_accuracy': 88.14,
                'selective_accuracy': 95.15,
                'temperature': 10.0,
                'confidence_threshold': 0.35
            }
        }
    
    def suggest_bias_mitigation_strategies(self, all_analyses):
        """
        Suggest strategies to mitigate detected biases.
        
        Based on the analysis results, recommends concrete actions
        to reduce or eliminate identified biases.
        
        Args:
            all_analyses (dict): Results from all bias analyses.
            
        Returns:
            list: Recommended mitigation strategies.
        """""
        print("\n BIAS MITIGATION STRATEGIES")
        print("="*60)
        
        strategies = []
        
        # Representation bias strategies
        if all_analyses.get('representation'):
            cv = all_analyses['representation']['cv']
            if cv > 0.1:
                strategies.append({
                    'type': 'Representation',
                    'strategy': 'Rebalance dataset',
                    'description': f'CV={cv:.3f} indicates imbalance. Use data augmentation or resampling.'
                })
        
        # Geographic bias strategies
        if all_analyses.get('geographical'):
            biases = all_analyses['geographical']['biases_detected']
            if any('EUROPEO' in bias for bias in biases):
                strategies.append({
                    'type': 'Geographic',
                    'strategy': 'Regional diversification',
                    'description': 'Include more breeds from Asia, Africa, and Americas for global balance.'
                })
        
        # Architectural bias strategies
        if all_analyses.get('architectural'):
            strategies.extend([
                {
                    'type': 'Architectural',
                    'strategy': 'Unified architectures',
                    'description': 'Use the same architecture (e.g., ResNet50) for all models.'
                },
                {
                    'type': 'Architectural',
                    'strategy': 'Single multi-head model',
                    'description': 'Replace hybrid system with a single model with multiple outputs.'
                },
                {
                    'type': 'Architectural',
                    'strategy': 'Selective model removal',
                    'description': 'Remove unfair advantage of the 6 breeds with specialized model.'
                }
            ])
        
        # Evaluation bias strategies
        strategies.extend([
            {
                'type': 'Evaluation',
                'strategy': 'Per-class metrics',
                'description': 'Report precision, recall, and F1-score for each individual breed.'
            },
            {
                'type': 'Evaluation',
                'strategy': 'Stratified test dataset',
                'description': 'Ensure balanced representation in the test set.'
            },
            {
                'type': 'Evaluation',
                'strategy': 'Adaptive thresholds',
                'description': 'Use breed-specific confidence thresholds based on performance.'
            },
            {
                'type': 'Evaluation',
                'strategy': 'Stratified cross-validation',
                'description': 'Use stratified k-fold for more robust evaluation.'
            }
        ])
        
        print(" RECOMMENDED STRATEGIES:")
        for i, strategy in enumerate(strategies, 1):
            print(f"\n{i:2d}. [{strategy['type']}] {strategy['strategy']}")
            print(f"     {strategy['description']}")
        
        return strategies
    
    def create_bias_visualization(self, all_analyses):
        """
        Create comprehensive bias analysis visualizations.
        
        Generates a multi-panel figure showing distributions and
        bias metrics across all analyzed dimensions.
        
        Args:
            all_analyses (dict): Results from all bias analyses.
            
        Returns:
            matplotlib.figure.Figure: Generated visualization figure.
        """
        print("\n CREATING BIAS VISUALIZATIONS...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # 1. Regional distribution
        if all_analyses.get('geographical'):
            regional_data = all_analyses['geographical']['regional_distribution']
            if regional_data:
                regions = list(regional_data.keys())
                counts = [len(breeds) for breeds in regional_data.values()]
                
                ax = axes[0]
                bars = ax.bar(regions, counts, color='lightblue', edgecolor='navy')
                ax.set_title('Geographic Distribution of Breeds', fontweight='bold')
                ax.set_ylabel('Number of Breeds')
                ax.tick_params(axis='x', rotation=45)
                
                # Add values on bars
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
        
        # 2. Popularity distribution
        if all_analyses.get('geographical'):
            popularity_data = all_analyses['geographical']['popularity_distribution']
            if popularity_data:
                popularities = list(popularity_data.keys())
                counts = [len(breeds) for breeds in popularity_data.values()]
                
                ax = axes[1]
                colors = ['red', 'orange', 'yellow', 'lightgreen'][:len(popularities)]
                bars = ax.bar(popularities, counts, color=colors, edgecolor='black')
                ax.set_title('Breed Popularity Distribution', fontweight='bold')
                ax.set_ylabel('Number of Breeds')
                
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
        
        # 3. Size distribution
        if all_analyses.get('physical'):
            size_data = all_analyses['physical']['size_distribution']
            if size_data:
                sizes = list(size_data.keys())
                counts = [len(breeds) for breeds in size_data.values()]
                
                ax = axes[2]
                bars = ax.bar(sizes, counts, color='lightcoral', edgecolor='darkred')
                ax.set_title('Breed Size Distribution', fontweight='bold')
                ax.set_ylabel('Number of Breeds')
                
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
        
        # 4. Balance of the dataset
        if all_analyses.get('representation'):
            breed_stats = all_analyses['representation']['breed_stats']
            if breed_stats:
                counts = list(breed_stats.values())
                ax = axes[3]
                ax.hist(counts, bins=15, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
                ax.set_title('Image Distribution per Breed', fontweight='bold')
                ax.set_xlabel('Number of Images')
                ax.set_ylabel('Number of Breeds')
                ax.axvline(np.mean(counts), color='red', linestyle='--', 
                          label=f'Media: {np.mean(counts):.0f}')
                ax.legend()
        
        # 5. Selective vs main breeds accuracy
        ax = axes[4]
        categories = ['Main Model\n(44 breeds)', 'Selective Model\n(6 breeds)']
        accuracies = [88.14, 95.15]
        colors = ['lightblue', 'orange']
        
        bars = ax.bar(categories, accuracies, color=colors, edgecolor='black')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(80, 100)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Bias summary
        ax = axes[5]
        ax.axis('off')
        
        # Count biases by category
        bias_summary = {
            'Geographic': len(all_analyses.get('geographical', {}).get('biases_detected', [])),
            'Physical': len(all_analyses.get('physical', {}).get('physical_biases', [])),
            'Architectural': len(all_analyses.get('architectural', {}).get('architectural_biases', [])),
            'Evaluation': len(all_analyses.get('evaluation', {}).get('evaluation_biases', []))
        }
        
        summary_text = " DETECTED BIASES SUMMARY\n\n"
        total_biases = 0
        for category, count in bias_summary.items():
            summary_text += f"{category}: {count} biases\n"
            total_biases += count
        
        summary_text += f"\nTotal: {total_biases} biases detected"
        
        if total_biases == 0:
            summary_text += "\n\n BIAS-FREE MODEL"
            color = 'green'
        elif total_biases < 5:
            summary_text += "\n\n MINOR BIASES"
            color = 'orange'
        else:
            summary_text += "\n\n SIGNIFICANT BIASES"
            color = 'red'
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.tight_layout()
        plt.savefig('bias_analysis_report.png', dpi=300, bbox_inches='tight')
        print("    Visualization saved: bias_analysis_report.png")
        
        return fig
    
    def run_complete_bias_analysis(self):
        """
        Run the complete bias analysis pipeline.
        
        Executes all bias analyses, generates visualizations,
        and saves comprehensive reports.
        
        Returns:
            dict: Complete analysis results, strategies, and visualizations.
        """
        print(" COMPLETE MODEL BIAS ANALYSIS")
        print("="*80)
        
        all_analyses = {}
        
        # 1. Representation bias
        all_analyses['representation'] = self.analyze_dataset_representation_bias()
        
        # 2. Geographical bias
        all_analyses['geographical'] = self.analyze_geographical_bias()
        
        # 3. Physical characteristics bias
        all_analyses['physical'] = self.analyze_physical_characteristics_bias()
        
        # 4. Architectural bias
        all_analyses['architectural'] = self.analyze_model_architecture_bias()
        
        # 5. Evaluation bias
        all_analyses['evaluation'] = self.analyze_evaluation_bias()
        
        # 6. Mitigation strategies
        strategies = self.suggest_bias_mitigation_strategies(all_analyses)
        
        # 7. Create visualizations
        fig = self.create_bias_visualization(all_analyses)
        
        # Save reporte complete
        self.save_bias_report(all_analyses, strategies)
        
        return {
            'analyses': all_analyses,
            'strategies': strategies,
            'visualization': fig
        }
    
    def save_bias_report(self, analyses, strategies):
        """
        Save comprehensive bias analysis report to files.
        
        Generates JSON report with all findings, critical issues,
        and recommendations for bias mitigation.
        
        Args:
            analyses (dict): Results from all bias analyses.
            strategies (list): Recommended mitigation strategies.
            
        Returns:
            dict: Complete report structure.
        """
        print("\n SAVING BIAS REPORT...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_info': {
                'architecture': 'Hierarchical (ResNet18 + ResNet50 + ResNet34)',
                'total_breeds': 50,
                'dataset_balance': analyses.get('representation', {}).get('balance_status', 'Unknown')
            },
            'bias_analyses': analyses,
            'mitigation_strategies': strategies,
            'summary': {
                'total_biases_detected': sum([
                    len(analyses.get('geographical', {}).get('biases_detected', [])),
                    len(analyses.get('physical', {}).get('physical_biases', [])),
                    len(analyses.get('architectural', {}).get('architectural_biases', [])),
                    len(analyses.get('evaluation', {}).get('evaluation_biases', []))
                ]),
                'critical_issues': [],
                'recommendations': []
            }
        }
        
        # Identify critical issues
        if analyses.get('representation', {}).get('cv', 0) > 0.2:
            report['summary']['critical_issues'].append('Dataset significantly imbalanced')
            
        if any('EUROPEAN' in bias.upper() for bias in analyses.get('geographical', {}).get('biases_detected', [])):
            report['summary']['critical_issues'].append('Geographic bias towards European breeds')
            
        # Main recommendations
        report['summary']['recommendations'] = [
            'Unify model architecture to eliminate unfair advantages',
            'Diversify dataset with more non-European breeds',
            'Implement per-class evaluation metrics',
            'Use stratified cross-validation for robust evaluation'
        ]
        
        # Save como JSON
        with open('bias_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print("    Reporte guardado: bias_analysis_report.json")
        
        return report

def main():
    """
    Main entry point for bias detection analysis.
    
    Initializes the analyzer and runs the complete analysis pipeline.
    
    Returns:
        dict: Complete analysis results.
    """
    workspace_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
    
    analyzer = BiasDetectionAnalyzer(workspace_path)
    results = analyzer.run_complete_bias_analysis()
    
    return results

if __name__ == "__main__":
    results = main()