"""
Breed Distribution and Training Performance Analyzer
=====================================================

This module analyzes the dog breed dataset distribution and estimates
training performance implications. Key analyses include:

- Breed image count distribution
- Class imbalance metrics and ratios
- Training time and complexity estimates
- Hardware requirements assessment
- Optimization strategy recommendations

Generates visualizations and statistics to guide training decisions.

Author: Dog Breed Classifier Team
Date: 2024
"""

import os
import time
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class BreedPerformanceAnalyzer:
    """
    Analyzer for breed dataset distribution and training performance.
    
    Provides comprehensive analysis of dataset balance, estimates training
    complexity, and recommends optimization strategies based on hardware.
    
    Attributes:
        yesdog_path (Path): Path to dog breed images.
        nodog_path (Path): Path to non-dog images.
        breed_stats (dict): Statistics for each breed.
    """
    
    def __init__(self, yesdog_path: str, nodog_path: str):
        """
        Initialize the breed performance analyzer.
        
        Args:
            yesdog_path (str): Path to the YESDOG directory.
            nodog_path (str): Path to the NODOG directory.
        """
        self.yesdog_path = Path(yesdog_path)
        self.nodog_path = Path(nodog_path)
        self.breed_stats = {}
        
    def analyze_breeds_distribution(self):
        """
        Analyze the distribution of images across dog breeds.
        
        Counts images per breed directory and provides summary statistics.
        
        Returns:
            tuple: (breed_counts dict, total_dog_images, nodog_images)
        """
        print("🔍 ANALYZING DOG BREEDS...")
        print("="*60)
        
        # Count images per breed
        breed_counts = {}
        total_dog_images = 0
        
        for breed_dir in self.yesdog_path.iterdir():
            if breed_dir.is_dir():
                # Contar files of image
                image_files = [f for f in breed_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                count = len(image_files)
                breed_counts[breed_dir.name] = count
                total_dog_images += count
        
        # Contar images NO-DOG
        nodog_images = 0
        for nodog_dir in self.nodog_path.iterdir():
            if nodog_dir.is_dir():
                image_files = [f for f in nodog_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                nodog_images += len(image_files)
        
        print(f"📊 ESTADÍSTICAS GENERALES:")
        print(f"   🐕 Razas de perros: {len(breed_counts)}")
        print(f"   🐕 Total imágenes de perros: {total_dog_images:,}")
        print(f"   ❌ Total imágenes NO-DOG: {nodog_images:,}")
        print(f"   📈 Total clases: {len(breed_counts) + 1} (120 razas + NO-DOG)")
        
        return breed_counts, total_dog_images, nodog_images
    
    def analyze_class_imbalance(self, breed_counts: dict, nodog_images: int):
        """
        Analyze class imbalance in the dataset.
        
        Calculates distribution statistics and identifies classes
        with significantly fewer images than average.
        
        Args:
            breed_counts (dict): Image counts per breed.
            nodog_images (int): Number of non-dog images.
            
        Returns:
            tuple: (all_counts, min_count, max_count, mean_count)
        """
        print(f"\n⚖️  CLASS IMBALANCE ANALYSIS:")
        print("="*60)
        
        # Include NO-DOG class in analysis
        all_counts = breed_counts.copy()
        all_counts['NO-DOG'] = nodog_images
        
        counts = list(all_counts.values())
        class_names = list(all_counts.keys())
        
        # Calculate statistics
        min_count = min(counts)
        max_count = max(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        print(f"   📉 Clase con menos imágenes: {min_count}")
        print(f"   📈 Clase con más imágenes: {max_count}")
        print(f"   📊 Promedio por clase: {mean_count:.1f}")
        print(f"   📏 Desviación estándar: {std_count:.1f}")
        print(f"   ⚠️  Ratio max/min: {max_count/min_count:.1f}x")
        
        # Encontrar classes with pocas images
        low_count_threshold = mean_count * 0.3  # 30% of the average
        low_count_classes = [(name, count) for name, count in all_counts.items() 
                           if count < low_count_threshold]
        
        if low_count_classes:
            print(f"\n⚠️  CLASES CON POCAS IMÁGENES (< {low_count_threshold:.0f}):")
            for name, count in sorted(low_count_classes, key=lambda x: x[1]):
                print(f"      {name}: {count} imágenes")
        
        return all_counts, min_count, max_count, mean_count
    
    def estimate_training_performance(self, total_classes: int, total_images: int):
        """
        Estimate training performance implications.
        
        Provides estimates for training time, memory requirements,
        and convergence difficulty based on dataset characteristics.
        
        Args:
            total_classes (int): Number of classes to train.
            total_images (int): Total images in the dataset.
            
        Returns:
            tuple: (estimated_time_per_epoch, complexity_factor)
        """
        print(f"\n🚀 TRAINING PERFORMANCE ANALYSIS:")
        print("="*60)
        
        # Compare binary vs multi-class complexity
        print("📊 BINARY vs MULTI-CLASS COMPARISON:")
        print(f"   Current model (binary): 2 classes")
        print(f"   Proposed model: {total_classes} classes")
        print(f"   Complexity factor: {total_classes/2:.0f}x")
        
        # Memory impact
        print(f"\n💾 MEMORY IMPACT:")
        # Assuming EfficientNet-B3
        base_params = 12_000_000  # ~12M base parameters
        
        # Final layer for classification
        binary_final_params = 1536 * 2  # 1536 features → 2 classes
        multiclass_final_params = 1536 * total_classes  # 1536 features → N classes
        
        print(f"   Binary final layer: {binary_final_params:,} parameters")
        print(f"   Multi-class final layer: {multiclass_final_params:,} parameters")
        print(f"   Increase: {multiclass_final_params - binary_final_params:,} parameters")
        print(f"   Memory increase: ~{(multiclass_final_params - binary_final_params) * 4 / 1024 / 1024:.1f} MB")
        
        # Training time estimates
        print(f"\n⏱️  TRAINING TIME:")
        print(f"   Total images: {total_images:,}")
        
        # Estimates based on experience
        base_time_per_epoch = total_images / 1000  # ~1000 images per minute
        complexity_factor = 1 + (total_classes - 2) * 0.02  # 2% increase per additional class
        
        estimated_time_per_epoch = base_time_per_epoch * complexity_factor
        
        print(f"   Estimated time per epoch: {estimated_time_per_epoch:.1f} minutes")
        print(f"   For 30 epochs: {estimated_time_per_epoch * 30:.1f} minutes (~{estimated_time_per_epoch * 30 / 60:.1f} hours)")
        
        # Dificultad of convergencia
        print(f"\n🎯 DIFICULTAD DE CONVERGENCIA:")
        if total_classes <= 10:
            difficulty = "FÁCIL"
            epochs_needed = "15-25"
        elif total_classes <= 50:
            difficulty = "MODERADO"
            epochs_needed = "25-40"
        elif total_classes <= 100:
            difficulty = "DIFÍCIL"
            epochs_needed = "40-60"
        else:
            difficulty = "MUY DIFÍCIL"
            epochs_needed = "50-80"
            
        print(f"   Dificultad: {difficulty}")
        print(f"   Épocas recomendadas: {epochs_needed}")
        print(f"   Razón: Con {total_classes} clases, el modelo necesita aprender")
        print(f"          muchas más características distintivas")
        
        return estimated_time_per_epoch, complexity_factor
    
    def recommend_optimization_strategies(self, breed_counts: dict, min_count: int, max_count: int):
        """
        Recommend optimization strategies based on dataset analysis.
        
        Provides actionable recommendations for data handling, model
        configuration, training approach, and hardware utilization.
        
        Args:
            breed_counts (dict): Image counts per breed.
            min_count (int): Minimum images in any class.
            max_count (int): Maximum images in any class.
        """
        print(f"\n💡 ESTRATEGIAS DE OPTIMIZACIÓN RECOMENDADAS:")
        print("="*60)
        
        print("1️⃣  ESTRATEGIAS DE DATOS:")
        if max_count / min_count > 10:
            print("   ⚖️  Balanceo agresivo necesario (ratio > 10x)")
            print("      - Undersample clases grandes a max 2000 imágenes")
            print("      - Oversample clases pequeñas (augmentación)")
            print("      - Usar weighted sampling durante entrenamiento")
        else:
            print("   ⚖️  Balanceo moderado suficiente")
            print("      - Weighted loss function")
            print("      - Augmentación ligera para clases pequeñas")
        
        print(f"\n2️⃣  ESTRATEGIAS DE MODELO:")
        print("   🧠 Transfer Learning OBLIGATORIO")
        print("      - ImageNet pre-entrenado es esencial")
        print("      - Freeze inicial de 10-15 épocas")
        print("      - Fine-tuning gradual")
        
        print(f"\n3️⃣  ESTRATEGIAS DE ENTRENAMIENTO:")
        print("   📈 Learning Rate Schedule:")
        print("      - OneCycleLR o CosineAnnealingLR")
        print("      - LR inicial: 1e-4 (más conservador)")
        print("      - Warmup de 5 épocas")
        
        print(f"\n4️⃣  ESTRATEGIAS DE HARDWARE:")
        print("   💻 Para AMD 7900XTX:")
        print("      - Batch size: 16-32 (por memoria)")
        print("      - Mixed precision (AMP)")
        print("      - Gradient accumulation si es necesario")
        
        print(f"\n5️⃣  ESTRATEGIAS DE VALIDACIÓN:")
        print("   📊 Métricas específicas:")
        print("      - Top-1 y Top-5 accuracy")
        print("      - F1-score por clase")
        print("      - Confusion matrix para clases problemáticas")
    
    def create_breed_visualization(self, breed_counts: dict, nodog_images: int):
        """
        Create comprehensive breed distribution visualizations.
        
        Generates multi-panel figure with top/bottom breeds, distribution
        histogram, and summary statistics.
        
        Args:
            breed_counts (dict): Image counts per breed.
            nodog_images (int): Number of non-dog images.
            
        Returns:
            pd.DataFrame: DataFrame with all breed counts.
        """
        print(f"\n📊 CREATING VISUALIZATIONS...")
        
        # Preparar data
        all_counts = breed_counts.copy()
        all_counts['NO-DOG'] = nodog_images
        
        # Create DataFrame
        df = pd.DataFrame(list(all_counts.items()), columns=['Breed', 'Count'])
        df = df.sort_values('Count', ascending=False)
        
        # Create figura with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Top 20 breeds
        top_20 = df.head(20)
        sns.barplot(data=top_20, x='Count', y='Breed', ax=ax1, palette='viridis')
        ax1.set_title('Top 20 Razas con Más Imágenes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Número de Imágenes')
        
        # 2. Bottom 20 breeds (excluyendo NO-DOG)
        bottom_20 = df[df['Breed'] != 'NO-DOG'].tail(20)
        sns.barplot(data=bottom_20, x='Count', y='Breed', ax=ax2, palette='rocket')
        ax2.set_title('Top 20 Razas con Menos Imágenes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Número de Imágenes')
        
        # 3. Distribution histogram
        ax3.hist(df['Count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Image Distribution per Class', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Images')
        ax3.set_ylabel('Number of Classes')
        ax3.axvline(df['Count'].mean(), color='red', linestyle='--', label=f'Average: {df["Count"].mean():.0f}')
        ax3.legend()
        
        # 4. Statistics summary
        ax4.axis('off')
        stats_text = f"""
DATASET STATISTICS

Total classes: {len(df)}
Total images: {df['Count'].sum():,}

Per class:
• Minimum: {df['Count'].min()} images
• Maximum: {df['Count'].max():,} images  
• Average: {df['Count'].mean():.0f} images
• Median: {df['Count'].median():.0f} images

Imbalance:
• Max/min ratio: {df['Count'].max()/df['Count'].min():.1f}x
• Std. deviation: {df['Count'].std():.0f}

Classes with < 100 images: {len(df[df['Count'] < 100])}
Classes with > 1000 images: {len(df[df['Count'] > 1000])}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('breed_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: breed_analysis.png")
        
        return df
    
    def run_complete_analysis(self):
        """
        Run the complete breed analysis pipeline.
        
        Executes all analysis components and generates comprehensive
        report with statistics, estimates, and recommendations.
        
        Returns:
            dict: Complete analysis results.
        """
        start_time = time.time()
        
        print("🐕 COMPLETE BREED AND PERFORMANCE ANALYSIS")
        print("="*80)
        
        # 1. Analyze breed distribution
        breed_counts, total_dog_images, nodog_images = self.analyze_breeds_distribution()
        
        # 2. Analizar desbalanceo
        all_counts, min_count, max_count, mean_count = self.analyze_class_imbalance(
            breed_counts, nodog_images
        )
        
        # 3. Estimar performance
        total_classes = len(breed_counts) + 1
        total_images = total_dog_images + nodog_images
        time_per_epoch, complexity_factor = self.estimate_training_performance(
            total_classes, total_images
        )
        
        # 4. Recomendar optimizaciones
        self.recommend_optimization_strategies(breed_counts, min_count, max_count)
        
        # 5. Create visualizaciones
        df = self.create_breed_visualization(breed_counts, nodog_images)
        
        # Resumen final
        elapsed_time = time.time() - start_time
        print(f"\n🎯 RESUMEN EJECUTIVO:")
        print("="*60)
        print(f"✅ Factibilidad del proyecto: ALTA")
        print(f"⚠️  Complejidad: ALTA (121 clases)")
        print(f"⏱️  Tiempo estimado de entrenamiento: {time_per_epoch * 50:.0f} minutos")
        print(f"💾 Memoria adicional requerida: ~{(121-2) * 1536 * 4 / 1024 / 1024:.1f} MB")
        print(f"🎯 Accuracy esperada: 75-85% (top-1), 90-95% (top-5)")
        
        print(f"\n📋 RECOMENDACIÓN:")
        if max_count / min_count > 20:
            print("   🔴 PRECAUCIÓN: Desbalanceo muy alto")
            print("   👉 Implementar balanceo agresivo antes de entrenar")
        elif total_classes > 100:
            print("   🟡 COMPLEJIDAD ALTA pero manejable")
            print("   👉 Usar transfer learning + estrategias de optimización")
        else:
            print("   🟢 FACTIBLE con estrategias estándar")
        
        print(f"\n⏱️  Análisis completado en {elapsed_time:.1f} segundos")
        
        return {
            'breed_counts': breed_counts,
            'total_classes': total_classes,
            'total_images': total_images,
            'time_per_epoch': time_per_epoch,
            'complexity_factor': complexity_factor,
            'imbalance_ratio': max_count / min_count,
            'dataframe': df
        }

def main():
    """
    Main entry point for breed performance analysis.
    
    Initializes the analyzer with dataset paths and runs complete analysis.
    
    Returns:
        dict: Complete analysis results.
    """
    yesdog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\YESDOG"
    nodog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\NODOG"
    
    analyzer = BreedPerformanceAnalyzer(yesdog_path, nodog_path)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()