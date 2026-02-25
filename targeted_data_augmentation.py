# !/usr/bin/env python3
"""
Technical documentation in English.
=====================================================

Technical documentation in English.
Technical documentation in English.

Autor: System IA
Fecha: 2024
"""

import os
import json
import numpy as np
import cv2
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import random

class TargetedDataAugmenter:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.datasets_path = self.workspace_path / "DATASETS"
        self.yesdog_path = self.datasets_path / "YESDOG"
        self.output_path = self.workspace_path / "BALANCED_AUGMENTED_DATASET"
        
        # Implementation note.
        self.load_problematic_classes()
        
        # Implementation note.
        self.setup_augmentation_strategies()
        
    def load_problematic_classes(self):
        """Technical documentation in English."""
        eval_file = self.workspace_path / "complete_class_evaluation_report.json"
        
        self.problematic_classes = []
        self.class_accuracies = {}
        
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            class_details = results.get('class_details', {})
            for breed, details in class_details.items():
                accuracy = details['accuracy']
                self.class_accuracies[breed] = accuracy
                
                # Clasificar por nivel de problema
                if accuracy < 0.60:
                    self.problematic_classes.append((breed, 'CRITICO'))
                elif accuracy < 0.70:
                    self.problematic_classes.append((breed, 'ALTO'))
                elif accuracy < 0.80:
                    self.problematic_classes.append((breed, 'MEDIO'))
        else:
            # Implementation note.
            known_problematic = [
                ('Lhasa', 0.536),
                ('cairn', 0.586), 
                ('Siberian_husky', 0.621),
                ('whippet', 0.643),
                ('Australian_terrier', 0.690),
                ('Norfolk_terrier', 0.692),
                ('giant_schnauzer', 0.667),
                ('soft-coated_wheaten_terrier', 0.659)
            ]
            
            for breed, acc in known_problematic:
                self.class_accuracies[breed] = acc
                if acc < 0.60:
                    self.problematic_classes.append((breed, 'CRITICO'))
                elif acc < 0.70:
                    self.problematic_classes.append((breed, 'ALTO'))
                else:
                    self.problematic_classes.append((breed, 'MEDIO'))
        
        print(f"📋 Clases problemáticas identificadas: {len(self.problematic_classes)}")
        for breed, level in self.problematic_classes:
            print(f"   🚨 {breed}: {level} (acc: {self.class_accuracies.get(breed, 0.0):.3f})")
    
    def setup_augmentation_strategies(self):
        """Configura estrategias de augmentation por nivel de severidad"""
        
        # Implementation note.
        self.critical_augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=0.3, p=0.5)
            ], p=0.6),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
            ], p=0.8),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ImageCompression(quality_lower=85, p=0.5)
            ], p=0.6),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.4)
        ])
        
        # Augmentation ALTO (moderadamente agresivo)
        self.high_augmentation = A.Compose([
            A.RandomRotate90(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.7),
            A.OneOf([
                A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.4),
                A.GridDistortion(p=0.4),
                A.OpticalDistortion(distort_limit=0.2, p=0.4)
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.4)
            ], p=0.6),
            A.OneOf([
                A.Blur(blur_limit=2, p=0.4),
                A.MotionBlur(blur_limit=2, p=0.4),
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.4)
            ], p=0.4),
            A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.3)
        ])
        
        # Augmentation MEDIO (conservador)
        self.medium_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3)
            ], p=0.4),
            A.OneOf([
                A.Blur(blur_limit=1, p=0.3),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3)
            ], p=0.2),
            A.CoarseDropout(max_holes=2, max_height=16, max_width=16, p=0.2)
        ])
        
        # Implementation note.
        self.normal_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)
        ])
    
    def analyze_current_distribution(self):
        """Technical documentation in English."""
        print(f"\n📊 ANALIZANDO DISTRIBUCIÓN ACTUAL")
        print("="*50)
        
        class_counts = {}
        total_images = 0
        
        if not self.yesdog_path.exists():
            print(f"❌ No se encontró el directorio YESDOG: {self.yesdog_path}")
            return None
        
        for breed_dir in self.yesdog_path.iterdir():
            if breed_dir.is_dir():
                breed_name = breed_dir.name
                # Contar images (jpg, jpeg, png)
                image_files = list(breed_dir.glob("*.jpg")) + \
                             list(breed_dir.glob("*.jpeg")) + \
                             list(breed_dir.glob("*.png")) + \
                             list(breed_dir.glob("*.JPEG"))
                
                count = len(image_files)
                class_counts[breed_name] = count
                total_images += count
        
        # Implementation note.
        counts = list(class_counts.values())
        if counts:
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            min_count = min(counts)
            max_count = max(counts)
            
            print(f"📈 ESTADÍSTICAS ACTUALES:")
            print(f"   Total de clases: {len(class_counts)}")
            print(f"   Total de imágenes: {total_images:,}")
            print(f"   Promedio por clase: {mean_count:.1f}")
            print(f"   Desviación estándar: {std_count:.1f}")
            print(f"   Rango: {min_count} - {max_count}")
            print(f"   Coeficiente de variación: {std_count/mean_count:.3f}")
            
            # Identificar classes with pocas images
            target_count = max_count  # Implementation note.
            underrepresented = [(breed, count, target_count-count) 
                              for breed, count in class_counts.items() 
                              if count < target_count]
            
            underrepresented.sort(key=lambda x: x[1])  # Ordenar por cantidad actual
            
            print(f"\n🎯 OBJETIVO DE BALANCEO: {target_count} imágenes por clase")
            print(f"📉 Clases que necesitan augmentation: {len(underrepresented)}")
            
            if underrepresented:
                print(f"\n🚨 TOP 10 CLASES MÁS DESBALANCEADAS:")
                for i, (breed, current, needed) in enumerate(underrepresented[:10], 1):
                    problem_level = "NORMAL"
                    for prob_breed, level in self.problematic_classes:
                        if prob_breed == breed:
                            problem_level = level
                            break
                    
                    print(f"   {i:2d}. {breed:25} | Actual: {current:3d} | Necesita: +{needed:3d} | {problem_level}")
            
            return {
                'class_counts': class_counts,
                'target_count': target_count,
                'underrepresented': underrepresented,
                'stats': {
                    'mean': mean_count,
                    'std': std_count,
                    'min': min_count,
                    'max': max_count,
                    'total': total_images,
                    'classes': len(class_counts)
                }
            }
        
        return None
    
    def get_augmentation_strategy(self, breed_name, current_count, needed_count):
        """Technical documentation in English."""
        
        # Determinar nivel de problema
        problem_level = "NORMAL"
        for prob_breed, level in self.problematic_classes:
            if prob_breed == breed_name:
                problem_level = level
                break
        
        # Determinar intensidad de augmentation basada en:
        # 1. Nivel de problema (accuracy)
        # 2. Cantidad de images faltantes
        
        shortage_ratio = needed_count / max(current_count, 1)
        
        if problem_level == "CRITICO" or shortage_ratio > 3:
            return self.critical_augmentation, "CRÍTICO", 6  # Implementation note.
        elif problem_level == "ALTO" or shortage_ratio > 2:
            return self.high_augmentation, "ALTO", 4
        elif problem_level == "MEDIO" or shortage_ratio > 1.5:
            return self.medium_augmentation, "MEDIO", 3
        else:
            return self.normal_augmentation, "NORMAL", 2
    
    def augment_breed_images(self, breed_name, breed_path, target_count, current_count):
        """Technical documentation in English."""
        
        needed_count = target_count - current_count
        if needed_count <= 0:
            return 0
        
        # Obtener estrategia de augmentation
        augmentation, strategy_level, variations_per_image = self.get_augmentation_strategy(
            breed_name, current_count, needed_count
        )
        
        print(f"   🎯 {breed_name} | Actual: {current_count} | Objetivo: {target_count} | Estrategia: {strategy_level}")
        
        # Crear directory de salida
        output_breed_path = self.output_path / breed_name
        output_breed_path.mkdir(parents=True, exist_ok=True)
        
        # Copiar images originales
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG']
        original_images = []
        
        for ext in image_extensions:
            original_images.extend(list(breed_path.glob(ext)))
        
        copied_count = 0
        for img_path in original_images:
            try:
                shutil.copy2(img_path, output_breed_path / img_path.name)
                copied_count += 1
            except Exception as e:
                print(f"      ❌ Error copiando {img_path.name}: {e}")
        
        # Generar images augmentadas
        generated_count = 0
        images_to_augment = original_images * ((needed_count // len(original_images)) + 1)
        random.shuffle(images_to_augment)
        
        for i, img_path in enumerate(images_to_augment[:needed_count]):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Aplicar augmentation
                augmented = augmentation(image=image)
                aug_image = augmented['image']
                
                # Save image augmentada
                aug_filename = f"{img_path.stem}_aug_{i:04d}{img_path.suffix}"
                aug_path = output_breed_path / aug_filename
                
                aug_image_pil = Image.fromarray(aug_image)
                aug_image_pil.save(aug_path, quality=95)
                
                generated_count += 1
                
                if generated_count % 50 == 0:
                    print(f"      📈 Generadas {generated_count}/{needed_count} imágenes...")
                    
            except Exception as e:
                print(f"      ❌ Error augmentando {img_path.name}: {e}")
                continue
        
        final_count = copied_count + generated_count
        print(f"      ✅ Completado: {copied_count} originales + {generated_count} augmentadas = {final_count} total")
        
        return generated_count
    
    def create_balanced_dataset(self):
        """Crea un dataset balanced with data augmentation dirigido"""
        print(f"\n🎯 INICIANDO CREACIÓN DE DATASET BALANCEADO")
        print("="*70)
        
        # Implementation note.
        distribution_data = self.analyze_current_distribution()
        if not distribution_data:
            print("❌ No se pudo analizar la distribución actual")
            return None
        
        class_counts = distribution_data['class_counts']
        target_count = distribution_data['target_count']
        underrepresented = distribution_data['underrepresented']
        
        # Crear directory de salida
        if self.output_path.exists():
            print(f"🗑️ Limpiando directorio existente...")
            shutil.rmtree(self.output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Procesar cada class
        print(f"\n🚀 PROCESANDO {len(class_counts)} CLASES...")
        print("="*70)
        
        total_generated = 0
        successful_classes = 0
        
        for breed_name, current_count in class_counts.items():
            breed_path = self.yesdog_path / breed_name
            
            if not breed_path.exists():
                print(f"❌ Directorio no encontrado: {breed_name}")
                continue
            
            try:
                generated = self.augment_breed_images(
                    breed_name, breed_path, target_count, current_count
                )
                total_generated += generated
                successful_classes += 1
                
            except Exception as e:
                print(f"❌ Error procesando {breed_name}: {e}")
                continue
        
        # Verificar resultado final
        final_distribution = self.verify_balanced_dataset()
        
        # Resumen
        print(f"\n✅ DATASET BALANCEADO CREADO EXITOSAMENTE")
        print("="*70)
        print(f"   📁 Ubicación: {self.output_path}")
        print(f"   🎯 Clases procesadas: {successful_classes}/{len(class_counts)}")
        print(f"   📈 Imágenes generadas: {total_generated:,}")
        print(f"   📊 Objetivo por clase: {target_count}")
        
        if final_distribution:
            print(f"   ✅ Balance logrado: {final_distribution['balanced_classes']}/{final_distribution['total_classes']} clases")
            print(f"   📈 Total final: {final_distribution['total_images']:,} imágenes")
        
        return {
            'output_path': str(self.output_path),
            'target_count': target_count,
            'generated_images': total_generated,
            'successful_classes': successful_classes,
            'total_classes': len(class_counts),
            'final_distribution': final_distribution
        }
    
    def verify_balanced_dataset(self):
        """Technical documentation in English."""
        if not self.output_path.exists():
            return None
        
        class_counts = {}
        for breed_dir in self.output_path.iterdir():
            if breed_dir.is_dir():
                # Contar all las images
                image_files = list(breed_dir.glob("*.jpg")) + \
                             list(breed_dir.glob("*.jpeg")) + \
                             list(breed_dir.glob("*.png")) + \
                             list(breed_dir.glob("*.JPEG"))
                
                class_counts[breed_dir.name] = len(image_files)
        
        if not class_counts:
            return None
        
        counts = list(class_counts.values())
        target_count = max(counts)
        balanced_classes = sum(1 for count in counts if count >= target_count * 0.95)  # 95% of the objetivo
        
        return {
            'class_counts': class_counts,
            'target_count': target_count,
            'balanced_classes': balanced_classes,
            'total_classes': len(class_counts),
            'total_images': sum(counts),
            'mean_count': np.mean(counts),
            'std_count': np.std(counts)
        }
    
    def create_visualization_report(self, results):
        """Crea un reporte visual of the process de balanceo"""
        if not results or not results.get('final_distribution'):
            print("❌ No hay datos para crear visualización")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎯 REPORTE DE DATA AUGMENTATION DIRIGIDO', fontsize=16, fontweight='bold')
        
        final_dist = results['final_distribution']
        class_counts = final_dist['class_counts']
        
        # Implementation note.
        breeds = list(class_counts.keys())[:20]  # Top 20 for legibilidad
        counts = [class_counts[breed] for breed in breeds]
        
        bars1 = ax1.bar(range(len(breeds)), counts, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.axhline(y=final_dist['target_count'], color='red', linestyle='--', 
                   label=f"Objetivo: {final_dist['target_count']}")
        ax1.set_xticks(range(len(breeds)))
        ax1.set_xticklabels([breed.replace('_', ' ') for breed in breeds], 
                           rotation=45, ha='right', fontsize=8)
        ax1.set_title('📊 Distribución Final (Top 20 Clases)')
        ax1.set_ylabel('Número de Imágenes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Implementation note.
        all_counts = list(class_counts.values())
        ax2.hist(all_counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.axvline(final_dist['mean_count'], color='red', linestyle='-', 
                   label=f"Media: {final_dist['mean_count']:.0f}")
        ax2.axvline(final_dist['target_count'], color='orange', linestyle='--',
                   label=f"Objetivo: {final_dist['target_count']}")
        ax2.set_xlabel('Imágenes por Clase')
        ax2.set_ylabel('Número de Clases')
        ax2.set_title('📈 Distribución de Imágenes por Clase')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Implementation note.
        problematic_breeds = [breed for breed, level in self.problematic_classes]
        problematic_counts = [class_counts.get(breed, 0) for breed in problematic_breeds]
        problematic_levels = [level for breed, level in self.problematic_classes]
        
        colors_map = {'CRITICO': 'red', 'ALTO': 'orange', 'MEDIO': 'yellow'}
        colors3 = [colors_map.get(level, 'gray') for level in problematic_levels]
        
        bars3 = ax3.bar(range(len(problematic_breeds)), problematic_counts, 
                       color=colors3, alpha=0.7, edgecolor='black')
        ax3.axhline(y=final_dist['target_count'], color='green', linestyle='--',
                   label=f"Objetivo: {final_dist['target_count']}")
        ax3.set_xticks(range(len(problematic_breeds)))
        ax3.set_xticklabels([breed.replace('_', ' ') for breed in problematic_breeds], 
                           rotation=45, ha='right', fontsize=9)
        ax3.set_title('🚨 Clases Problemáticas Mejoradas')
        ax3.set_ylabel('Imágenes Finales')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Implementation note.
        stats_labels = ['Clases\nProcesadas', 'Imágenes\nGeneradas', 'Balance\nLogrado %']
        stats_values = [
            results['successful_classes'],
            results['generated_images'],
            (final_dist['balanced_classes'] / final_dist['total_classes']) * 100
        ]
        
        bars4 = ax4.bar(stats_labels, stats_values, 
                       color=['lightblue', 'lightcoral', 'lightgreen'], 
                       alpha=0.7, edgecolor='black')
        ax4.set_title('📊 Estadísticas de Mejora')
        ax4.set_ylabel('Valor')
        
        # Implementation note.
        for bar, value in zip(bars4, stats_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(stats_values)*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('targeted_augmentation_report.png', dpi=300, bbox_inches='tight')
        print("✅ Reporte visual guardado: targeted_augmentation_report.png")
        
        # Save reporte JSON
        report_data = {
            'timestamp': str(np.datetime64('now')),
            'results': results,
            'problematic_classes': dict(self.problematic_classes),
            'class_accuracies': self.class_accuracies
        }
        
        with open('targeted_augmentation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        print("✅ Reporte JSON guardado: targeted_augmentation_report.json")
    
    def run_complete_augmentation(self):
        """Ejecuta el process completo de augmentation dirigido"""
        print("🎯" * 70)
        print("🎯 DATA AUGMENTATION DIRIGIDO PARA RAZAS PROBLEMÁTICAS")
        print("🎯" * 70)
        
        try:
            # Crear dataset balanced
            results = self.create_balanced_dataset()
            
            if results:
                # Crear reporte visual
                self.create_visualization_report(results)
                
                print(f"\n🏆 PROCESO COMPLETADO EXITOSAMENTE")
                print(f"   📁 Dataset balanceado: {results['output_path']}")
                print(f"   📈 Imágenes generadas: {results['generated_images']:,}")
                print(f"   🎯 Clases balanceadas: {results['successful_classes']}/{results['total_classes']}")
                
                return results
            else:
                print("❌ Error en el proceso de augmentation")
                return None
                
        except Exception as e:
            print(f"❌ Error en el proceso completo: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Function principal"""
    workspace_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
    
    augmenter = TargetedDataAugmenter(workspace_path)
    results = augmenter.run_complete_augmentation()
    
    return results

if __name__ == "__main__":
    results = main()