#!/usr/bin/env python3
"""
Análisis de balance de clases en el dataset de 50 razas
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json

def analyze_breed_balance():
    """Analizar el balance de clases en el dataset de razas"""
    
    print("🔍 ANÁLISIS DE BALANCE DE CLASES - 50 RAZAS")
    print("=" * 60)
    
    train_dir = "breed_processed_data/train"
    if not os.path.exists(train_dir):
        print("❌ Directorio de entrenamiento no encontrado!")
        return
    
    # Contar imágenes por raza
    breed_counts = {}
    total_images = 0
    
    breeds = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    breeds.sort()
    
    print(f"📊 Encontradas {len(breeds)} razas:")
    print("-" * 60)
    
    for breed in breeds:
        breed_path = os.path.join(train_dir, breed)
        count = len([f for f in os.listdir(breed_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        breed_counts[breed] = count
        total_images += count
        print(f"   {breed}: {count:>4} imágenes")
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   Total de imágenes: {total_images:,}")
    print(f"   Promedio por raza: {total_images/len(breeds):.1f}")
    print(f"   Mínimo: {min(breed_counts.values())} ({min(breed_counts, key=breed_counts.get)})")
    print(f"   Máximo: {max(breed_counts.values())} ({max(breed_counts, key=breed_counts.get)})")
    print(f"   Desviación estándar: {np.std(list(breed_counts.values())):.1f}")
    
    # Análisis de desbalance
    counts = list(breed_counts.values())
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    cv = std_count / mean_count  # Coeficiente de variación
    
    print(f"\n⚖️ ANÁLISIS DE DESBALANCE:")
    print(f"   Coeficiente de variación: {cv:.3f}")
    if cv > 0.3:
        print("   🔴 DATASET SIGNIFICATIVAMENTE DESBALANCEADO")
    elif cv > 0.1:
        print("   🟡 DATASET MODERADAMENTE DESBALANCEADO") 
    else:
        print("   🟢 DATASET BIEN BALANCEADO")
    
    # Razas más desbalanceadas
    print(f"\n📊 TOP 10 RAZAS CON MÁS IMÁGENES:")
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (breed, count) in enumerate(sorted_breeds[:10], 1):
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>4} ({percentage:.1f}%)")
    
    print(f"\n📊 TOP 10 RAZAS CON MENOS IMÁGENES:")
    for i, (breed, count) in enumerate(sorted_breeds[-10:], 1):
        percentage = (count / total_images) * 100
        print(f"   {i:2d}. {breed}: {count:>4} ({percentage:.1f}%)")
    
    # Generar gráfico
    plt.figure(figsize=(15, 8))
    breeds_list = [breed.replace('_', ' ').title() for breed, _ in sorted_breeds]
    counts_list = [count for _, count in sorted_breeds]
    
    plt.bar(range(len(breeds_list)), counts_list, color='skyblue', alpha=0.7)
    plt.axhline(y=mean_count, color='red', linestyle='--', label=f'Promedio: {mean_count:.1f}')
    plt.axhline(y=mean_count + std_count, color='orange', linestyle=':', alpha=0.7, label=f'+1 SD: {mean_count + std_count:.1f}')
    plt.axhline(y=mean_count - std_count, color='orange', linestyle=':', alpha=0.7, label=f'-1 SD: {mean_count - std_count:.1f}')
    
    plt.xlabel('Razas (ordenadas por cantidad)')
    plt.ylabel('Número de imágenes')
    plt.title('Distribución de imágenes por raza de perro (50 razas)')
    plt.xticks(range(0, len(breeds_list), 5), 
               [breeds_list[i][:10] + '...' if len(breeds_list[i]) > 10 else breeds_list[i] 
                for i in range(0, len(breeds_list), 5)], 
               rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('breed_balance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Guardar resultados
    results = {
        'total_breeds': len(breeds),
        'total_images': total_images,
        'mean_images_per_breed': mean_count,
        'std_deviation': std_count,
        'coefficient_of_variation': cv,
        'min_images': min(counts),
        'max_images': max(counts),
        'breed_counts': breed_counts,
        'balance_status': 'desbalanceado' if cv > 0.1 else 'balanceado'
    }
    
    with open('breed_balance_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados guardados en:")
    print(f"   - breed_balance_report.json")
    print(f"   - breed_balance_analysis.png")
    
    return results

def recommend_balancing_strategy(results):
    """Recomendar estrategia de balanceado"""
    print(f"\n🎯 RECOMENDACIONES DE BALANCEADO:")
    
    cv = results['coefficient_of_variation']
    mean_count = results['mean_images_per_breed']
    
    if cv > 0.3:
        print("   🔴 Dataset muy desbalanceado - Se requiere balanceado agresivo:")
        print("      - Data augmentation para razas con pocas imágenes")
        print("      - Undersampling para razas con demasiadas imágenes")
        print("      - Weighted loss function en el entrenamiento")
        target_per_class = min(int(mean_count * 1.2), max(results['breed_counts'].values()))
    elif cv > 0.1:
        print("   🟡 Dataset moderadamente desbalanceado - Balanceado suave:")
        print("      - Data augmentation ligera")
        print("      - Class weights en la loss function")
        target_per_class = int(mean_count)
    else:
        print("   🟢 Dataset bien balanceado - Mantener como está")
        return results
    
    print(f"   📊 Objetivo recomendado: {target_per_class} imágenes por raza")
    
    # Calcular necesidades de balanceado
    breeds_need_more = []
    breeds_need_less = []
    
    for breed, count in results['breed_counts'].items():
        if count < target_per_class * 0.8:  # Menos del 80% del objetivo
            breeds_need_more.append((breed, count, target_per_class - count))
        elif count > target_per_class * 1.5:  # Más del 150% del objetivo
            breeds_need_less.append((breed, count, count - target_per_class))
    
    if breeds_need_more:
        print(f"\n📈 Razas que necesitan MÁS imágenes ({len(breeds_need_more)}):")
        for breed, current, needed in sorted(breeds_need_more, key=lambda x: x[2], reverse=True)[:10]:
            print(f"      {breed}: {current} → {target_per_class} (+{needed})")
    
    if breeds_need_less:
        print(f"\n📉 Razas que necesitan MENOS imágenes ({len(breeds_need_less)}):")
        for breed, current, excess in sorted(breeds_need_less, key=lambda x: x[2], reverse=True)[:10]:
            print(f"      {breed}: {current} → {target_per_class} (-{excess})")
    
    return {
        'target_per_class': target_per_class,
        'breeds_need_more': breeds_need_more,
        'breeds_need_less': breeds_need_less,
        'balancing_required': len(breeds_need_more) > 0 or len(breeds_need_less) > 0
    }

if __name__ == "__main__":
    # Asegurarse de que no ejecute el sistema principal
    import sys
    sys.path.insert(0, '.')
    
    results = analyze_breed_balance()
    if results:
        balancing = recommend_balancing_strategy(results)
        
        if balancing and balancing.get('balancing_required'):
            print(f"\n🔧 ¿Quieres proceder con el balanceado automático? (y/n)")
            # Para automatizar, asumimos 'y'
            response = 'y'
            if response.lower() == 'y':
                print("✅ Procediendo con balanceado automático...")
            else:
                print("⏹️ Balanceado cancelado por el usuario")