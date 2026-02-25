# !/usr/bin/env python3
"""
Technical documentation in English.
===============================================

Technical documentation in English.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def create_retraining_visualization():
    """Technical documentation in English."""
    
    # Load data of the reporte
    workspace_path = Path(r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG")
    eval_file = workspace_path / "complete_class_evaluation_report.json"
    
    if eval_file.exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        print("❌ No se encontraron datos de evaluación")
        return
    
    # Configurar estilo
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔬 ANÁLISIS DE NECESIDAD DE REENTRENAMIENTO', fontsize=16, fontweight='bold')
    
    # Implementation note.
    class_details = results.get('class_details', {})
    accuracies = [details['accuracy'] for details in class_details.values()]
    
    ax1.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axvline(np.mean(accuracies), color='red', linestyle='--', 
               label=f'Media: {np.mean(accuracies):.3f}')
    ax1.axvline(0.7, color='orange', linestyle='--', 
               label='Umbral problemático (0.70)', alpha=0.8)
    ax1.set_xlabel('Accuracy por Clase')
    ax1.set_ylabel('Número de Clases')
    ax1.set_title('📊 Distribución de Accuracy por Clase')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Implementation note.
    criteria_names = ['Accuracy\nPromedio', 'Variabilidad\nEntre Clases', 
                     'Brecha de\nRendimiento', 'Clases\nProblemáticas']
    current_values = [0.865, 0.119, 0.464, 8]
    thresholds = [0.85, 0.15, 0.30, 8]
    
    colors = ['green' if curr <= thresh else 'orange' 
              for curr, thresh in zip(current_values, thresholds)]
    
    bars = ax2.bar(criteria_names, current_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Implementation note.
    for i, thresh in enumerate(thresholds):
        ax2.axhline(y=thresh, xmin=i/len(thresholds), xmax=(i+1)/len(thresholds), 
                   color='red', linestyle='--', alpha=0.8)
    
    ax2.set_title('🎯 Criterios de Evaluación vs Umbrales')
    ax2.set_ylabel('Valor')
    ax2.grid(True, alpha=0.3)
    
    # Implementation note.
    for bar, value in zip(bars, current_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Implementation note.
    options = ['Mantener\nActual', 'Optimización\nActual', 
               'Fine-tuning\nDirigido', 'Reentrenamiento\nCompleto']
    accuracy_gains = [0.000, 0.020, 0.050, 0.080]
    time_costs = [0, 1, 3, 6]
    
    # Implementation note.
    sizes = [50, 100, 200, 400]  # Proporcional to the esfuerzo
    colors_cost = ['green', 'lightgreen', 'orange', 'red']
    
    scatter = ax3.scatter(time_costs, accuracy_gains, s=sizes, c=colors_cost, 
                         alpha=0.7, edgecolors='black', linewidths=2)
    
    # Labels
    for i, option in enumerate(options):
        ax3.annotate(option, (time_costs[i], accuracy_gains[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Tiempo de Implementación (semanas)')
    ax3.set_ylabel('Ganancia de Accuracy Esperada')
    ax3.set_title('💰 Análisis Costo-Beneficio')
    ax3.grid(True, alpha=0.3)
    
    # Implementation note.
    problematic_breeds = []
    problematic_accs = []
    
    for breed, details in class_details.items():
        if details['accuracy'] < 0.70:
            problematic_breeds.append(breed.replace('_', ' '))
            problematic_accs.append(details['accuracy'])
    
    # Ordenar for accuracy
    sorted_data = sorted(zip(problematic_breeds, problematic_accs), 
                        key=lambda x: x[1])
    
    if sorted_data:
        breeds, accs = zip(*sorted_data)
        
        bars4 = ax4.barh(range(len(breeds)), accs, color='lightcoral', 
                        alpha=0.7, edgecolor='darkred')
        ax4.set_yticks(range(len(breeds)))
        ax4.set_yticklabels(breeds, fontsize=9)
        ax4.axvline(x=0.7, color='orange', linestyle='--', 
                   label='Umbral problemático')
        ax4.set_xlabel('Accuracy')
        ax4.set_title('🚨 Clases Más Problemáticas')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Implementation note.
        for i, (bar, acc) in enumerate(zip(bars4, accs)):
            ax4.text(acc + 0.01, i, f'{acc:.3f}', 
                    va='center', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No hay clases\nproblemáticas', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, fontweight='bold')
        ax4.set_title('✅ Sin Clases Problemáticas')
    
    plt.tight_layout()
    plt.savefig('retraining_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Visualización guardada: retraining_analysis_visualization.png")
    
    # Implementation note.
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Implementation note.
    categories = ['Accuracy\nGeneral', 'Variabilidad\nBaja', 'Sin Brecha\nExcesiva', 'Pocas Clases\nProblemáticas']
    scores = [0.865/0.95, (0.15-0.119)/0.15, (0.40-0.464)/0.40, (12-8)/12]  # Normalizado a 0-1
    scores = [max(0, min(1, score)) for score in scores]  # Clamp entre 0-1
    
    # Implementation note.
    scores += scores[:1]
    categories += categories[:1]
    
    angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]
    
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, scores, 'o-', linewidth=2, label='Rendimiento Actual')
    ax.fill(angles, scores, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    ax.set_ylim(0, 1)
    ax.set_title('🎯 Evaluación Multidimensional del Modelo', 
                size=16, fontweight='bold', pad=20)
    
    # Implementation note.
    good_threshold = [0.8] * len(categories)
    ax.plot(angles, good_threshold, '--', color='green', alpha=0.7, 
           label='Umbral "Bueno"')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.savefig('retraining_recommendation_radar.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico de recomendación guardado: retraining_recommendation_radar.png")
    
    print(f"\n🎨 VISUALIZACIONES CREADAS:")
    print(f"   📊 retraining_analysis_visualization.png - Análisis completo")
    print(f"   🎯 retraining_recommendation_radar.png - Evaluación multidimensional")

if __name__ == "__main__":
    create_retraining_visualization()