#!/usr/bin/env python3
"""
🐕 ANÁLISIS ESPECÍFICO: MALTESE DOG
==================================
Análisis detallado del rendimiento del Maltese en el modelo de 119 clases
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_maltese_performance():
    """Análisis completo del Maltese Dog"""
    
    print("🐕 ANÁLISIS ESPECÍFICO: MALTESE DOG")
    print("=" * 50)
    
    # Cargar métricas
    try:
        with open('class_metrics.json', 'r') as f:
            class_metrics = json.load(f)
        
        with open('complete_class_evaluation_report.json', 'r') as f:
            complete_metrics = json.load(f)
            
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # Métricas del Maltese
    maltese_metrics = class_metrics.get('Maltese_dog', {})
    maltese_complete = complete_metrics.get('class_reports', {}).get('Maltese_dog', {})
    
    if not maltese_metrics:
        print("❌ No se encontraron métricas para Maltese_dog")
        return
    
    print("\n📊 MÉTRICAS DE RENDIMIENTO:")
    print("-" * 30)
    print(f"🎯 Accuracy:           {maltese_metrics.get('accuracy', 0):.1%}")
    print(f"🎯 Precision:          {maltese_metrics.get('precision', 0):.1%}")
    print(f"🎯 Recall:             {maltese_metrics.get('recall', 0):.1%}")
    print(f"🎯 F1-Score:           {maltese_metrics.get('f1_score', 0):.3f}")
    print(f"📋 Muestras evaluadas: {int(maltese_metrics.get('samples_evaluated', 0))}")
    print(f"📋 Support:            {int(maltese_metrics.get('support', 0))}")
    
    print("\n🔍 ANÁLISIS DE CONFIANZA:")
    print("-" * 30)
    avg_conf = maltese_metrics.get('avg_confidence', 0)
    std_conf = maltese_metrics.get('std_confidence', 0)
    min_conf = maltese_complete.get('min_confidence', 0)
    max_conf = maltese_complete.get('max_confidence', 0)
    
    print(f"📈 Confianza promedio: {avg_conf:.1%}")
    print(f"📊 Desv. estándar:     {std_conf:.4f}")
    print(f"📉 Confianza mínima:   {min_conf:.1%}")
    print(f"📈 Confianza máxima:   {max_conf:.1%}")
    
    # Comparación con otras razas
    all_f1_scores = [metrics.get('f1_score', 0) for metrics in class_metrics.values()]
    all_accuracies = [metrics.get('accuracy', 0) for metrics in class_metrics.values()]
    all_confidences = [metrics.get('avg_confidence', 0) for metrics in class_metrics.values()]
    
    maltese_f1 = maltese_metrics.get('f1_score', 0)
    maltese_accuracy = maltese_metrics.get('accuracy', 0)
    maltese_confidence = maltese_metrics.get('avg_confidence', 0)
    
    # Percentiles
    f1_percentile = (sum(1 for score in all_f1_scores if score < maltese_f1) / len(all_f1_scores)) * 100
    acc_percentile = (sum(1 for acc in all_accuracies if acc < maltese_accuracy) / len(all_accuracies)) * 100
    conf_percentile = (sum(1 for conf in all_confidences if conf < maltese_confidence) / len(all_confidences)) * 100
    
    print("\n📊 COMPARACIÓN CON OTRAS RAZAS:")
    print("-" * 35)
    print(f"🏆 F1-Score:    Top {100-f1_percentile:.0f}% (Percentil {f1_percentile:.0f})")
    print(f"🏆 Accuracy:    Top {100-acc_percentile:.0f}% (Percentil {acc_percentile:.0f})")  
    print(f"🏆 Confianza:   Top {100-conf_percentile:.0f}% (Percentil {conf_percentile:.0f})")
    
    # Análisis de sesgo específico
    print("\n🔍 ANÁLISIS DE SESGO ESPECÍFICO:")
    print("-" * 35)
    
    # Identificar razones del buen rendimiento
    reasons = []
    if maltese_f1 > 0.90:
        reasons.append("✅ Excelente F1-Score (>0.90)")
    if maltese_accuracy == 1.0:
        reasons.append("✅ Accuracy perfecta (100%)")
    if std_conf < 0.01:
        reasons.append("✅ Muy baja variabilidad en confianza")
    if avg_conf > 0.99:
        reasons.append("✅ Confianza promedio muy alta (>99%)")
    
    potential_issues = []
    if maltese_metrics.get('precision', 0) < maltese_metrics.get('recall', 0):
        potential_issues.append("⚠️ Precision menor que Recall (posibles falsos positivos)")
    if std_conf > 0.1:
        potential_issues.append("⚠️ Alta variabilidad en confianza")
    
    print("\n🟢 FORTALEZAS IDENTIFICADAS:")
    for reason in reasons:
        print(f"  {reason}")
    
    if potential_issues:
        print("\n🟡 POSIBLES ÁREAS DE MEJORA:")
        for issue in potential_issues:
            print(f"  {issue}")
    else:
        print("\n🟢 NO SE IDENTIFICARON PROBLEMAS SIGNIFICATIVOS")
    
    # Análisis de similitud con otras razas pequeñas
    small_dogs = ['toy_terrier', 'papillon', 'Japanese_spaniel', 'Pomeranian', 'Chihuahua']
    available_small_dogs = {name: metrics for name, metrics in class_metrics.items() 
                           if any(small in name.lower() for small in ['toy', 'papillon', 'japanese', 'pomeranian', 'chihuahua'])}
    
    print(f"\n🐕 COMPARACIÓN CON PERROS PEQUEÑOS SIMILARES:")
    print("-" * 45)
    print(f"{'Raza':25} | {'F1':6} | {'Acc':6} | {'Conf':6}")
    print("-" * 45)
    print(f"{'Maltese_dog':25} | {maltese_f1:.3f} | {maltese_accuracy:.3f} | {maltese_confidence:.3f}")
    
    for breed, metrics in available_small_dogs.items():
        f1 = metrics.get('f1_score', 0)
        acc = metrics.get('accuracy', 0)
        conf = metrics.get('avg_confidence', 0)
        print(f"{breed[:25]:25} | {f1:.3f} | {acc:.3f} | {conf:.3f}")
    
    # Conclusiones
    print("\n" + "="*50)
    print("📋 CONCLUSIONES SOBRE MALTESE DOG")
    print("="*50)
    
    if maltese_f1 > 0.90 and maltese_accuracy > 0.95:
        print("🎉 EXCELENTE RENDIMIENTO - Maltese muestra uno de los mejores desempeños")
        print("✅ Prácticamente sin sesgo detectado")
        print("🏆 Modelo muy confiable para esta raza")
        
        if std_conf < 0.01:
            print("🎯 Predicciones muy consistentes y confiables")
            
        if maltese_accuracy == 1.0:
            print("🥇 PERFECTO: 100% de accuracy en el conjunto de prueba")
    
    elif maltese_f1 > 0.80:
        print("👍 BUEN RENDIMIENTO - Maltese tiene un desempeño sólido")
        print("✅ Sesgo mínimo detectado")
    
    else:
        print("⚠️ RENDIMIENTO SUBÓPTIMO - Posible sesgo presente")
    
    print(f"\n💡 RECOMENDACIÓN: {'MANTENER modelo actual' if maltese_f1 > 0.90 else 'Considerar mejoras'}")
    
    return {
        'breed': 'Maltese_dog',
        'performance_level': 'EXCELENTE' if maltese_f1 > 0.90 else 'BUENO' if maltese_f1 > 0.80 else 'REGULAR',
        'bias_level': 'NINGUNO' if maltese_f1 > 0.90 and std_conf < 0.01 else 'BAJO',
        'metrics': maltese_metrics,
        'percentiles': {
            'f1': f1_percentile,
            'accuracy': acc_percentile,
            'confidence': conf_percentile
        }
    }

if __name__ == "__main__":
    result = analyze_maltese_performance()