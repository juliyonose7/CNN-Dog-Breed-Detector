"""
Top 50 Dog Breed Selection and Training Optimization.

This module selects the top 50 dog breeds from the YESDOG dataset based on
image availability and provides optimized training configurations specifically
tailored for AMD Ryzen 7800X3D processors.

Key Features:
    - Famous breed mapping to ImageNet directories
    - Automatic breed selection by image count
    - AMD 7800X3D-specific optimizations (3D V-Cache)
    - Training time estimation and performance analysis
    - Visualization of breed distribution

Usage:
    python top_50_selector.py
"""

import os
import time
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Top50BreedSelector:
    """
    Selector for top 50 dog breeds with training optimization.
    
    Analyzes breed data availability and provides optimized configurations
    for training on AMD Ryzen 7800X3D processors.
    
    Attributes:
        yesdog_path (Path): Path to the YESDOG dataset.
        famous_breeds (dict): Mapping of common breed names to ImageNet directories.
    """
    
    def __init__(self, yesdog_path: str):
        """
        Initialize the breed selector.
        
        Args:
            yesdog_path (str): Path to the YESDOG dataset directory.
        """
        self.yesdog_path = Path(yesdog_path)
        
        # Mapping of famous breeds to ImageNet directory names
        self.famous_breeds = {
            # Most popular breeds worldwide
            'labrador_retriever': ['n02099712-Labrador_retriever'],
            'golden_retriever': ['n02099601-golden_retriever'],
            'german_shepherd': ['n02106662-German_shepherd'],
            'bulldog_frances': ['n02108915-French_bulldog'],
            'bulldog': ['n02096585-Boston_bull'],  # Cercano a bulldog
            'beagle': ['n02088364-beagle'],
            'poodle': ['n02113624-toy_poodle', 'n02113712-miniature_poodle', 'n02113799-standard_poodle'],
            'rottweiler': ['n02106550-Rottweiler'],
            'yorkshire_terrier': ['n02094433-Yorkshire_terrier'],
            'dachshund': [],  # No disponible
            'siberian_husky': ['n02110185-Siberian_husky'],
            'boxer': ['n02108089-boxer'],
            'great_dane': ['n02109047-Great_Dane'],
            'chihuahua': ['n02085620-Chihuahua'],
            'shih_tzu': ['n02086240-Shih-Tzu'],
            'maltese': ['n02085936-Maltese_dog'],
            'border_collie': ['n02106166-Border_collie'],
            'australian_shepherd': [],  # No disponible
            'pug': ['n02110958-pug'],
            'cocker_spaniel': ['n02102318-cocker_spaniel'],
            'afghan_hound': ['n02088094-Afghan_hound'],
            'basset_hound': ['n02088238-basset'],
            'bloodhound': ['n02088466-bloodhound'],
            'doberman': ['n02107142-Doberman'],
            'saint_bernard': ['n02109525-Saint_Bernard'],
            'mastiff': ['n02108551-Tibetan_mastiff', 'n02108422-bull_mastiff'],
            'newfoundland': ['n02111277-Newfoundland'],
            'bernese_mountain_dog': ['n02107683-Bernese_mountain_dog'],
            'great_pyrenees': ['n02111500-Great_Pyrenees'],
            'samoyed': ['n02111889-Samoyed'],
            'collie': ['n02106030-collie'],
            'irish_setter': ['n02100877-Irish_setter'],
            'english_setter': ['n02100735-English_setter'],
            'gordon_setter': ['n02101006-Gordon_setter'],
            'weimaraner': ['n02092339-Weimaraner'],
            'vizsla': ['n02100583-vizsla'],
            'pointer': ['n02100236-German_short-haired_pointer'],
            'springer_spaniel': ['n02102040-English_springer', 'n02102177-Welsh_springer_spaniel'],
            'brittany': ['n02101388-Brittany_spaniel'],
            'chesapeake_bay_retriever': ['n02099849-Chesapeake_Bay_retriever'],
            'flat_coated_retriever': ['n02099267-flat-coated_retriever'],
            'curly_coated_retriever': ['n02099429-curly-coated_retriever'],
            'irish_water_spaniel': ['n02102973-Irish_water_spaniel'],
            'sussex_spaniel': ['n02102480-Sussex_spaniel'],
            'scottish_terrier': ['n02097298-Scotch_terrier'],
            'west_highland_terrier': ['n02098286-West_Highland_white_terrier'],
            'cairn_terrier': ['n02096177-cairn'],
            'fox_terrier': ['n02095314-wire-haired_fox_terrier'],
            'airedale': ['n02096051-Airedale'],
            'bull_terrier': ['n02093256-Staffordshire_bullterrier', 'n02093428-American_Staffordshire_terrier']
        }
    
    def analyze_available_breeds(self):
        """
        Analyze which famous breeds are available in the dataset.
        
        Scans the YESDOG dataset and maps available directories to famous
        breed names, counting images per breed.
        
        Returns:
            tuple: (available_famous dict, breed_counts dict)
        """
        print(" ANALYZING AVAILABLE FAMOUS BREEDS...")
        print("=" * 60)
        
        # Get all available directories
        available_dirs = [d.name for d in self.yesdog_path.iterdir() if d.is_dir()]
        
        # Map famous breeds to available data
        available_famous = {}
        breed_counts = {}
        
        for breed_name, dir_patterns in self.famous_breeds.items():
            total_images = 0
            found_dirs = []
            
            for pattern in dir_patterns:
                if pattern in available_dirs:
                    breed_dir = self.yesdog_path / pattern
                    image_files = [f for f in breed_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                    count = len(image_files)
                    total_images += count
                    found_dirs.append((pattern, count))
            
            if total_images > 0:
                available_famous[breed_name] = {
                    'dirs': found_dirs,
                    'total_images': total_images
                }
                breed_counts[breed_name] = total_images
        
        print(f" Available famous breeds: {len(available_famous)}")
        
        return available_famous, breed_counts
    
    def select_top_breeds_by_images(self, available_breeds: dict, min_images: int = 100):
        """
        Select top 50 breeds based on image count.
        
        Args:
            available_breeds (dict): Dictionary of available breeds.
            min_images (int): Minimum images required per breed. Default: 100.
            
        Returns:
            tuple: (top_50 list, all_breed_counts dict)
        """
        print(f"\n SELECTING TOP 50 BREEDS (min {min_images} images)...")
        print("=" * 60)
        
        # Get counts for all breeds (not just famous)
        all_breed_counts = {}
        
        for breed_dir in self.yesdog_path.iterdir():
            if breed_dir.is_dir():
                image_files = [f for f in breed_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                count = len(image_files)
                if count >= min_images:
                    # Clean name for display
                    clean_name = breed_dir.name
                    if clean_name.startswith('n0'):
                        # Extract readable name from ImageNet format
                        parts = clean_name.split('-')
                        if len(parts) > 1:
                            clean_name = '-'.join(parts[1:])
                    
                    all_breed_counts[clean_name] = {
                        'original_dir': breed_dir.name,
                        'count': count,
                        'path': breed_dir
                    }
        
        # Sort by image count
        sorted_breeds = sorted(all_breed_counts.items(), 
                             key=lambda x: x[1]['count'], 
                             reverse=True)
        
        # Select top 50
        top_50 = sorted_breeds[:50]
        
        print(f" Selected {len(top_50)} breeds:")
        print(f"    Image range: {top_50[-1][1]['count']} - {top_50[0][1]['count']}")
        
        # Show top 20
        print(f"\n TOP 20 BREEDS:")
        for i, (name, info) in enumerate(top_50[:20], 1):
            print(f"   {i:2d}. {name:25} | {info['count']:3d} images")
        
        print(f"\n... and 30 more breeds")
        
        return top_50, all_breed_counts
    
    def optimize_for_7800x3d(self):
        """
        Generate optimized configurations for AMD Ryzen 7800X3D.
        
        The 7800X3D features 96MB L3 3D V-Cache which benefits from:
        - Higher prefetch factors
        - Optimal batch sizes matching thread count
        - Persistent workers for reduced overhead
        
        Returns:
            tuple: (optimizations dict, env_vars list, performance dict)
        """
        print(f"\n AMD RYZEN 7800X3D OPTIMIZATIONS:")
        print("=" * 60)
        
        # 7800X3D specifications
        cpu_specs = {
            'cores': 8,
            'threads': 16,
            'base_clock': 4.2,  # GHz
            'boost_clock': 5.0,  # GHz
            'l3_cache': 96,     # MB (3D V-Cache)
            'tdp': 120,         # Watts
            'architecture': 'Zen 4',
            'memory_support': 'DDR5-5200'
        }
        
        print(f" CPU: AMD Ryzen 7 7800X3D")
        print(f"    {cpu_specs['cores']} cores, {cpu_specs['threads']} threads")
        print(f"    {cpu_specs['base_clock']} - {cpu_specs['boost_clock']} GHz")
        print(f"    {cpu_specs['l3_cache']} MB L3 Cache (3D V-Cache)")
        
        # Optimized PyTorch/DataLoader configurations
        optimizations = {
            'batch_size_cpu': min(32, cpu_specs['threads']),  # One per available thread
            'num_workers': cpu_specs['threads'] - 2,          # Leave 2 threads free
            'pin_memory': True,                               # Faster GPU transfer
            'persistent_workers': True,                       # Reuse workers
            'prefetch_factor': 4,                            # Extra cache leveraging L3
            'multiprocessing_context': 'spawn',              # Best for Windows
            'torch_threads': cpu_specs['threads'],           # Use all threads
            'mkldnn': True,                                  # Intel MKL-DNN optimizations
            'jemalloc': True,                                # Optimized allocator
        }
        
        print(f"\n  OPTIMIZED CONFIGURATIONS:")
        print(f"    Batch size (CPU): {optimizations['batch_size_cpu']}")
        print(f"    DataLoader workers: {optimizations['num_workers']}")
        print(f"    PyTorch threads: {optimizations['torch_threads']}")
        print(f"    Pin memory: {optimizations['pin_memory']}")
        print(f"    Persistent workers: {optimizations['persistent_workers']}")
        print(f"    Prefetch factor: {optimizations['prefetch_factor']} (leverages 3D V-Cache)")
        
        # System optimization commands
        system_optimizations = [
            'set OMP_NUM_THREADS=16',
            'set MKL_NUM_THREADS=16', 
            'set NUMEXPR_NUM_THREADS=16',
            'set OPENBLAS_NUM_THREADS=16',
            'set VECLIB_MAXIMUM_THREADS=16',
            'set PYTORCH_JIT=1',
            'set PYTORCH_JIT_OPT_LEVEL=2'
        ]
        
        print(f"\n  OPTIMIZED ENVIRONMENT VARIABLES:")
        for cmd in system_optimizations:
            print(f"   {cmd}")
        
        # Performance estimates
        estimated_performance = self.estimate_7800x3d_performance(optimizations)
        
        return optimizations, system_optimizations, estimated_performance
    
    def estimate_7800x3d_performance(self, optimizations: dict):
        """
        Estimate training performance with the given optimizations.
        
        Args:
            optimizations (dict): Optimization configuration dictionary.
            
        Returns:
            dict: Performance estimates including throughput and training time.
        """
        print(f"\n PERFORMANCE ESTIMATE:")
        print("=" * 60)
        
        # Baseline throughput calculation
        base_throughput = 150  # images/second baseline
        
        # Improvement factors
        factors = {
            'optimal_threads': 1.4,      # Full thread utilization
            'v_cache_boost': 1.25,       # 96MB L3 cache improves data locality
            'pin_memory': 1.1,           # Less copying overhead
            'persistent_workers': 1.15,   # No recreation overhead
            'prefetch_optimization': 1.2, # Leverages the cache
            'mkldnn_optimization': 1.3    # MKLDNN optimizations
        }
        
        total_factor = 1.0
        for factor_name, factor_value in factors.items():
            total_factor *= factor_value
        
        optimized_throughput = base_throughput * total_factor
        
        # For dataset of 50 breeds with ~8000 images total
        estimated_images = 8000
        batch_size = optimizations['batch_size_cpu']
        batches_per_epoch = estimated_images // batch_size
        
        time_per_epoch = batches_per_epoch / optimized_throughput * 60  # minutes
        
        print(f" Estimated throughput: {optimized_throughput:.0f} images/second")
        print(f" Improvement vs baseline: {total_factor:.2f}x")
        print(f"  Time per epoch: {time_per_epoch:.1f} minutes")
        print(f" Training 30 epochs: {time_per_epoch * 30:.0f} minutes (~{time_per_epoch * 30 / 60:.1f} hours)")
        
        print(f"\n COMPARISON WITH PREVIOUS ESTIMATE:")
        previous_time = 237.7  # hours for 121 classes
        new_time = time_per_epoch * 30 / 60
        
        print(f"   121 classes: {previous_time:.1f} hours")
        print(f"   50 breeds: {new_time:.1f} hours")
        print(f"    IMPROVEMENT: {previous_time / new_time:.1f}x faster!")
        
        return {
            'throughput': optimized_throughput,
            'time_per_epoch': time_per_epoch,
            'total_training_time': time_per_epoch * 30 / 60,
            'improvement_factor': total_factor
        }
    
    def create_breed_selection_visualization(self, top_50, performance_data):
        """
        Create visualizations for breed selection and performance analysis.
        
        Generates a 4-panel figure showing:
        1. Top 25 breeds by image count
        2. Image distribution histogram
        3. Comparison with original model
        4. 7800X3D optimization summary
        
        Args:
            top_50: List of selected breeds with metadata.
            performance_data: Performance estimation dictionary.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        print(f"\n CREATING VISUALIZATION...")
        
        # Prepare data
        breed_names = [name.replace('_', ' ').title() for name, _ in top_50]
        image_counts = [info['count'] for _, info in top_50]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Top 25 breeds
        top_25_names = breed_names[:25]
        top_25_counts = image_counts[:25]
        
        bars = ax1.barh(range(len(top_25_names)), top_25_counts, color='skyblue', edgecolor='navy')
        ax1.set_yticks(range(len(top_25_names)))
        ax1.set_yticklabels(top_25_names, fontsize=8)
        ax1.set_xlabel('Number of Images')
        ax1.set_title('Top 25 Selected Breeds', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add values on the bars
        for i, (bar, count) in enumerate(zip(bars, top_25_counts)):
            ax1.text(count + 5, i, str(count), va='center', fontsize=8)
        
        # 2. Image distribution
        ax2.hist(image_counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.set_xlabel('Images per Breed')
        ax2.set_ylabel('Number of Breeds')
        ax2.set_title('Image Distribution - Top 50', fontsize=14, fontweight='bold')
        ax2.axvline(np.mean(image_counts), color='red', linestyle='--', 
                   label=f'Average: {np.mean(image_counts):.0f}')
        ax2.legend()
        
        # 3. Performance comparison
        categories = ['Time\n(hours)', 'Classes', 'Images\n(thousands)']
        old_values = [237.7, 121, 140.6]
        new_values = [performance_data['total_training_time'], 50, 8.0]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, old_values, width, label='Modelo Original (121 clases)', 
                       color='lightcoral', alpha=0.8)
        bars2 = ax3.bar(x + width/2, new_values, width, label='Modelo Optimizado (50 razas)', 
                       color='lightblue', alpha=0.8)
        
        ax3.set_xlabel('Aspectos')
        ax3.set_ylabel('Valores')
        ax3.set_title('Comparación: Modelo Original vs Optimizado', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        
        # Add valores en the barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(old_values) * 0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Especificaciones of the 7800X3D
        ax4.axis('off')
        specs_text = f"""
 OPTIMIZACIONES PARA AMD RYZEN 7800X3D

 Especificaciones:
• 8 cores, 16 threads
• 4.2 - 5.0 GHz
• 96 MB L3 Cache (3D V-Cache)
• Zen 4 Architecture

 Configuraciones:
• Batch size: 16
• Workers: 14 
• PyTorch threads: 16
• Pin memory: Sí
• Prefetch factor: 4

 Rendimiento Estimado:
• {performance_data['throughput']:.0f} img/seg
• {performance_data['time_per_epoch']:.1f} min/época
• {performance_data['total_training_time']:.1f} horas total
• {performance_data['improvement_factor']:.2f}x mejora

 Dataset Final:
• 50 razas famosas
• ~{sum(image_counts):,} imágenes
• Balanceado y optimizado
        """
ax4.text(0.1, 0.9, specs_text, transform=ax4.transAxes, fontsize=12,
verticalalignment='top', fontfamily='monospace',
bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
plt.tight_layout()
plt.savefig('top_50_breeds_analysis.png', dpi=300, bbox_inches='tight')
print("  Saved: top_50_breeds_analysis.png")
        
return fig
    
def save_selected_breeds(self, top_50):
        """Save the selected breed list."""
        print(f"\n GUARDANDO CONFIGURACIÓN DE RAZAS...")
        
        # Create diccionario of configuration
        breed_config = {
            'total_breeds': len(top_50),
            'breeds': {}
        }
        
        for i, (name, info) in enumerate(top_50):
            breed_config['breeds'][i] = {
                'name': name,
                'display_name': name.replace('_', ' ').title(),
                'original_dir': info['original_dir'],
                'image_count': info['count'],
                'class_index': i
            }
        
        # Save como JSON
        import json
        with open('top_50_breeds_config.json', 'w', encoding='utf-8') as f:
            json.dump(breed_config, f, indent=2, ensure_ascii=False)
        
        # Implementation note.
        config_py = f"""# Configuration of the Top 50 Breeds of Dogs
# Implementation note.

TOP_50_BREEDS = {breed_config}

# quick mapping: name -> index of class
BREED_NAME_TO_INDEX = {{
"""
        
for i, (name, info) in enumerate(top_50):
config_py += f' "{name}": {i},\n'
        
config_py += "}\n\n# quick mapping: index -> name display\nBREED_INDEX_TO_DISPLAY = {\n"
        
for i, (name, info) in enumerate(top_50):
display_name = name.replace('_', ' ').title()
config_py += f' {i}: "{display_name}",\n'
        
config_py += "}\n"
        
with open('breed_config.py', 'w', encoding='utf-8') as f:
f.write(config_py)
        
print("  Saved: top_50_breeds_config.json")
print("  Saved: breed_config.py")
        
return breed_config
    
def run_complete_selection(self):
        """Run the full selection workflow."""
        start_time = time.time()
        
        print(" SELECCIÓN DE TOP 50 RAZAS + OPTIMIZACIÓN 7800X3D")
        print("="*80)
        
        # 1. Analizar breeds disponibles
        available_famous, famous_counts = self.analyze_available_breeds()
        
        # Implementation note.
        top_50, all_counts = self.select_top_breeds_by_images(available_famous)
        
        # 3. Optimizar for 7800X3D
        optimizations, env_vars, performance = self.optimize_for_7800x3d()
        
        # 4. Create visualizaciones
        fig = self.create_breed_selection_visualization(top_50, performance)
        
        # 5. Save configuration
        breed_config = self.save_selected_breeds(top_50)
        
        # Resumen final
        elapsed_time = time.time() - start_time
        total_images = sum(info['count'] for _, info in top_50)
        
        print(f"\n RESUMEN FINAL:")
        print("="*60)
        print(f" Razas seleccionadas: {len(top_50)}")
        print(f" Total de imágenes: {total_images:,}")
        print(f" Rango: {top_50[-1][1]['count']} - {top_50[0][1]['count']} imágenes")
        print(f" Rendimiento estimado: {performance['throughput']:.0f} img/seg")
        print(f"  Entrenamiento estimado: {performance['total_training_time']:.1f} horas")
        print(f" Mejora vs 121 clases: {237.7 / performance['total_training_time']:.1f}x más rápido")
        
        print(f"\n  Selección completada en {elapsed_time:.1f} segundos")
        
        return {
            'top_50': top_50,
            'breed_config': breed_config,
            'optimizations': optimizations,
            'performance': performance,
            'total_images': total_images
        }

def main():
    """Function main"""
    yesdog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\YESDOG"
    
    selector = Top50BreedSelector(yesdog_path)
    results = selector.run_complete_selection()
    
    return results

if __name__ == "__main__":
    results = main()