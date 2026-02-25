"""
Technical documentation in English.
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
    def __init__(self, yesdog_path: str, nodog_path: str):
        self.yesdog_path = Path(yesdog_path)
        self.nodog_path = Path(nodog_path)
        self.breed_stats = {}
        
    def analyze_breeds_distribution(self):
        """Technical documentation in English."""
        print("🔍 ANALIZANDO RAZAS DE PERROS...")
        print("="*60)
        
        # Contar images for breed
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
        """Analiza the desbalanceo of classes"""
        print(f"\n⚖️  ANÁLISIS DE DESBALANCEO:")
        print("="*60)
        
        # Implementation note.
        all_counts = breed_counts.copy()
        all_counts['NO-DOG'] = nodog_images
        
        counts = list(all_counts.values())
        class_names = list(all_counts.keys())
        
        # Implementation note.
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
        """Estima the consecuencias of performance of the training"""
        print(f"\n🚀 ANÁLISIS DE RENDIMIENTO DE ENTRENAMIENTO:")
        print("="*60)
        
        # Implementation note.
        print("📊 COMPARACIÓN BINARIO vs MULTI-CLASE:")
        print(f"   Modelo actual (binario): 2 clases")
        print(f"   Modelo propuesto: {total_classes} clases")
        print(f"   Factor de complejidad: {total_classes/2:.0f}x")
        
        # Memoria of the model
        print(f"\n💾 IMPACTO EN MEMORIA:")
        # Suponiendo EfficientNet-B3
        base_params = 12_000_000  # Implementation note.
        
        # Layer final for classification
        binary_final_params = 1536 * 2  # 1536 features → 2 classes
        multiclass_final_params = 1536 * total_classes  # 1536 features → 121 classes
        
        print(f"   Capa final binaria: {binary_final_params:,} parámetros")
        print(f"   Capa final multi-clase: {multiclass_final_params:,} parámetros")
        print(f"   Incremento: {multiclass_final_params - binary_final_params:,} parámetros")
        print(f"   Incremento memoria: ~{(multiclass_final_params - binary_final_params) * 4 / 1024 / 1024:.1f} MB")
        
        # Time of training
        print(f"\n⏱️  TIEMPO DE ENTRENAMIENTO:")
        print(f"   Imágenes totales: {total_images:,}")
        
        # Estimaciones basadas en experiencia
        base_time_per_epoch = total_images / 1000  # ~1000 images for minuto
        complexity_factor = 1 + (total_classes - 2) * 0.02  # Implementation note.
        
        estimated_time_per_epoch = base_time_per_epoch * complexity_factor
        
        print(f"   Tiempo estimado por época: {estimated_time_per_epoch:.1f} minutos")
        print(f"   Para 30 épocas: {estimated_time_per_epoch * 30:.1f} minutos (~{estimated_time_per_epoch * 30 / 60:.1f} horas)")
        
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
        """Recomienda estrategias of optimization"""
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
        """Technical documentation in English."""
        print(f"\n📊 CREANDO VISUALIZACIONES...")
        
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
        
        # Implementation note.
        ax3.hist(df['Count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribución de Imágenes por Clase', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Número de Imágenes')
        ax3.set_ylabel('Número de Clases')
        ax3.axvline(df['Count'].mean(), color='red', linestyle='--', label=f'Promedio: {df["Count"].mean():.0f}')
        ax3.legend()
        
        # Implementation note.
        ax4.axis('off')
        stats_text = f"""
ESTADÍSTICAS DEL DATASET

Total de clases: {len(df)}
Total de imágenes: {df['Count'].sum():,}

Por clase:
• Mínimo: {df['Count'].min()} imágenes
• Máximo: {df['Count'].max():,} imágenes  
• Promedio: {df['Count'].mean():.0f} imágenes
• Mediana: {df['Count'].median():.0f} imágenes

Desbalanceo:
• Ratio max/min: {df['Count'].max()/df['Count'].min():.1f}x
• Desv. estándar: {df['Count'].std():.0f}

Clases con < 100 imágenes: {len(df[df['Count'] < 100])}
Clases con > 1000 imágenes: {len(df[df['Count'] > 1000])}
        """
ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
verticalalignment='top', fontfamily='monospace',
bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
plt.tight_layout()
plt.savefig('breed_analysis.png', dpi=300, bbox_inches='tight')
print(" ✅ Saved: breed_analysis.png")
        
return df
    
def run_complete_analysis(self):
        """Run the full analysis pipeline."""
        start_time = time.time()
        
        print("🐕 ANÁLISIS COMPLETO DE RAZAS Y RENDIMIENTO")
        print("="*80)
        
        # Implementation note.
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
    """Technical documentation in English."""
    yesdog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\YESDOG"
    nodog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\NODOG"
    
    analyzer = BreedPerformanceAnalyzer(yesdog_path, nodog_path)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()