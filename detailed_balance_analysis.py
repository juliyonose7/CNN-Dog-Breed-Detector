# !/usr/bin/env python3
"""
Technical documentation in English.
"""

import os
import numpy as np
import json
from collections import Counter

def detailed_balance_analysis():
    """Technical documentation in English."""
    
    print("🔍 ANÁLISIS DETALLADO DE BALANCE DE CLASES")
    print("=" * 60)
    
    train_dir = "breed_processed_data/train"
    breeds = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    # Contar images for breed
    breed_counts = {}
    for breed in breeds:
        breed_path = os.path.join(train_dir, breed)
        count = len([f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        breed_counts[breed] = count
    
    # Implementation note.
    counts = list(breed_counts.values())
    total_images = sum(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    min_count = min(counts)
    max_count = max(counts)
    cv = std_count / mean_count
    
    print(f"📊 ESTADÍSTICAS GENERALES:")
    print(f"   Total de razas: {len(breeds)}")
    print(f"   Total de imágenes: {total_images:,}")
    print(f"   Promedio por raza: {mean_count:.1f}")
    print(f"   Desviación estándar: {std_count:.1f}")
    print(f"   Coeficiente de variación: {cv:.3f}")
    print(f"   Rango: {min_count} - {max_count} ({max_count - min_count} diferencia)")
    
    # Clasificar the nivel of desbalance
    if cv > 0.5:
        balance_status = "🔴 SEVERAMENTE DESBALANCEADO"
        priority = "CRÍTICA"
    elif cv > 0.3:
        balance_status = "🟠 FUERTEMENTE DESBALANCEADO"
        priority = "ALTA"
    elif cv > 0.1:
        balance_status = "🟡 MODERADAMENTE DESBALANCEADO"
        priority = "MEDIA"
    else:
        balance_status = "🟢 BIEN BALANCEADO"
        priority = "BAJA"
    
    print(f"\n⚖️ EVALUACIÓN DE BALANCE:")
    print(f"   Estado: {balance_status}")
    print(f"   Prioridad de corrección: {priority}")
    
    # Implementation note.
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1])
    
    print(f"\n📉 TOP 10 RAZAS CON MENOS IMÁGENES:")
    for i, (breed, count) in enumerate(sorted_breeds[:10], 1):
        deficit = mean_count - count
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>3} imágenes ({percentage:.1f}%, déficit: {deficit:+.0f})")
    
    print(f"\n📈 TOP 10 RAZAS CON MÁS IMÁGENES:")
    for i, (breed, count) in enumerate(sorted_breeds[-10:], 1):
        excess = count - mean_count
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>3} imágenes ({percentage:.1f}%, exceso: {excess:+.0f})")
    
    # Implementation note.
    q1 = np.percentile(counts, 25)
    q2 = np.percentile(counts, 50)  # mediana
    q3 = np.percentile(counts, 75)
    iqr = q3 - q1
    
    print(f"\n📊 ANÁLISIS DE CUARTILES:")
    print(f"   Q1 (25%): {q1:.1f}")
    print(f"   Q2 (50%, mediana): {q2:.1f}")
    print(f"   Q3 (75%): {q3:.1f}")
    print(f"   IQR: {iqr:.1f}")
    
    # Detect outliers
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr
    
    outliers_low = [(breed, count) for breed, count in breed_counts.items() if count < outlier_threshold_low]
    outliers_high = [(breed, count) for breed, count in breed_counts.items() if count > outlier_threshold_high]
    
    if outliers_low or outliers_high:
        print(f"\n🚨 OUTLIERS DETECTADOS:")
        if outliers_low:
            print(f"   Razas con muy pocas imágenes (< {outlier_threshold_low:.1f}):")
            for breed, count in sorted(outliers_low, key=lambda x: x[1]):
                print(f"      - {breed}: {count}")
        
        if outliers_high:
            print(f"   Razas con demasiadas imágenes (> {outlier_threshold_high:.1f}):")
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
    """Technical documentation in English."""
    
    print(f"\n🎯 ESTRATEGIA DE BALANCEADO RECOMENDADA")
    print("=" * 60)
    
    stats = analysis['stats']
    cv = stats['cv']
    mean_count = stats['mean']
    
    # Definir target of balanced
    if cv > 0.3:
        # For datasets very desbalanceados, use the mediana como target
        target_count = int(stats['q2'])
        strategy = "AGRESIVA"
    else:
        # For moderadamente desbalanceados, use average ajustado
        target_count = int(mean_count)
        strategy = "CONSERVADORA"
    
    print(f"📋 PARÁMETROS DE BALANCEADO:")
    print(f"   Estrategia: {strategy}")
    print(f"   Objetivo por raza: {target_count} imágenes")
    print(f"   Rango aceptable: {int(target_count * 0.9)} - {int(target_count * 1.1)}")
    
    # Calcular acciones necesarias
    breeds_to_augment = []  # Implementation note.
    breeds_to_reduce = []   # Necesitan less images
    breeds_balanced = []    # Implementation note.
    
    for breed, count in analysis['breed_counts'].items():
        if count < target_count * 0.9:
            needed = target_count - count
            breeds_to_augment.append((breed, count, needed))
        elif count > target_count * 1.1:
            excess = count - target_count
            breeds_to_reduce.append((breed, count, excess))
        else:
            breeds_balanced.append((breed, count))
    
    print(f"\n📈 RAZAS QUE NECESITAN AUMENTAR ({len(breeds_to_augment)}):")
    total_augmentation_needed = 0
    for breed, current, needed in sorted(breeds_to_augment, key=lambda x: x[2], reverse=True):
        print(f"   {breed}: {current} → {target_count} (+{needed})")
        total_augmentation_needed += needed
    
    print(f"\n📉 RAZAS QUE NECESITAN REDUCIR ({len(breeds_to_reduce)}):")
    total_reduction_possible = 0
    for breed, current, excess in sorted(breeds_to_reduce, key=lambda x: x[2], reverse=True):
        print(f"   {breed}: {current} → {target_count} (-{excess})")
        total_reduction_possible += excess
    
    print(f"\n✅ RAZAS YA BALANCEADAS ({len(breeds_balanced)}):")
    for breed, count in breeds_balanced:
        print(f"   {breed}: {count} ✓")
    
    print(f"\n📊 RESUMEN DE ACCIONES:")
    print(f"   Total imágenes a generar: {total_augmentation_needed}")
    print(f"   Total imágenes a reducir: {total_reduction_possible}")
    print(f"   Balance neto: {total_augmentation_needed - total_reduction_possible:+d}")
    
    # Implementation note.
    print(f"\n🔧 TÉCNICAS RECOMENDADAS:")
    
    if len(breeds_to_augment) > 0:
        print("   Para aumentar imágenes:")
        print("      - Data Augmentation (rotación, flip, zoom)")
        print("      - Generación sintética con GANs")
        print("      - Web scraping supervisado")
        print("      - Transfer learning desde razas similares")
    
    if len(breeds_to_reduce) > 0:
        print("   Para reducir imágenes:")
        print("      - Random sampling estratificado")
        print("      - Mantener solo las imágenes de mejor calidad")
        print("      - Preservar diversidad dentro de cada raza")
    
    print("   Otras recomendaciones:")
    print("      - Weighted loss function en el entrenamiento")
    print("      - Class-balanced sampling durante training")
    print("      - Validation set estratificado")
    
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
    print("Ejecutando análisis de balance detallado...")
    
    # Implementation note.
    analysis = detailed_balance_analysis()
    
    # Strategy of balanced
    strategy = propose_balancing_strategy(analysis)
    
    # Save resultados
    results = {
        'analysis': analysis,
        'strategy': strategy
    }
    
    with open('detailed_balance_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Reporte detallado guardado en: detailed_balance_report.json")