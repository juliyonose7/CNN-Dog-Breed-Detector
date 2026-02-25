# !/usr/bin/env python3
"""
Technical documentation in English.
==========================================

Technical documentation in English.
Technical documentation in English.

Autor: System IA
Fecha: 2024
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class RetrainingAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        
        # Implementation note.
        self.load_previous_results()
    
    def load_previous_results(self):
        """Load resultados of evaluaciones previas"""
        self.bias_analysis = {}
        self.class_evaluation = {}
        
        # Implementation note.
        bias_file = self.workspace_path / "bias_analysis_report.json"
        if bias_file.exists():
            with open(bias_file, 'r', encoding='utf-8') as f:
                self.bias_analysis = json.load(f)
        
        # Implementation note.
        class_file = self.workspace_path / "complete_class_evaluation_report.json"
        if class_file.exists():
            with open(class_file, 'r', encoding='utf-8') as f:
                self.class_evaluation = json.load(f)
    
    def analyze_current_performance_gaps(self):
        """Technical documentation in English."""
        print("🔍 ANÁLISIS DE BRECHAS DE RENDIMIENTO ACTUALES")
        print("="*70)
        
        if not self.class_evaluation:
            print("❌ No se encontraron resultados de evaluación por clase")
            return None
        
        # Implementation note.
        class_details = self.class_evaluation.get('class_details', {})
        problematic_classes = []
        excellent_classes = []
        
        for breed, details in class_details.items():
            accuracy = details.get('accuracy', 0.0)
            if accuracy < 0.7:
                problematic_classes.append((breed, accuracy))
            elif accuracy > 0.95:
                excellent_classes.append((breed, accuracy))
        
        # Implementation note.
        accuracies = [details['accuracy'] for details in class_details.values()]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        
        print(f"📊 ESTADÍSTICAS DE RENDIMIENTO ACTUAL:")
        print(f"   Accuracy promedio: {mean_acc:.3f}")
        print(f"   Desviación estándar: {std_acc:.3f}")
        print(f"   Rango: {min_acc:.3f} - {max_acc:.3f}")
        print(f"   Clases problemáticas (<0.70): {len(problematic_classes)}")
        print(f"   Clases excelentes (>0.95): {len(excellent_classes)}")
        
        # Calcular brecha of performance
        performance_gap = max_acc - min_acc
        print(f"   🚨 BRECHA DE RENDIMIENTO: {performance_gap:.3f}")
        
        # Implementation note.
        needs_retraining = self._evaluate_retraining_need(
            mean_acc, std_acc, len(problematic_classes), performance_gap
        )
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'performance_gap': performance_gap,
            'problematic_classes': problematic_classes,
            'excellent_classes': excellent_classes,
            'needs_retraining': needs_retraining
        }
    
    def _evaluate_retraining_need(self, mean_acc, std_acc, problematic_count, gap):
        """Technical documentation in English."""
        reasons = []
        priority = "LOW"
        
        # Criterios for retraining
        if std_acc > 0.15:
            reasons.append(f"Alta variabilidad entre clases (std={std_acc:.3f})")
            priority = "MEDIUM"
        
        if gap > 0.4:
            reasons.append(f"Brecha excesiva entre mejor/peor clase ({gap:.3f})")
            priority = "HIGH"
        
        if problematic_count > 8:
            reasons.append(f"Demasiadas clases problemáticas ({problematic_count})")
            priority = "HIGH"
        
        if mean_acc < 0.85:
            reasons.append(f"Accuracy promedio baja ({mean_acc:.3f})")
            priority = "MEDIUM"
        
        return {
            'recommended': len(reasons) > 0,
            'priority': priority,
            'reasons': reasons
        }
    
    def categorize_improvement_strategies(self):
        """Categoriza estrategias of improvement for if requieren retraining or no"""
        print("\n🔧 CATEGORIZACIÓN DE ESTRATEGIAS DE MEJORA")
        print("="*70)
        
        # Mejoras without retraining (ya implementadas)
        no_retraining = {
            "✅ IMPLEMENTADAS SIN REENTRENAMIENTO": [
                "Eliminar modelo selectivo (arquitectura unificada)",
                "Umbrales adaptativos por raza", 
                "Métricas detalladas por clase individual",
                "Calibración de temperatura optimizada",
                "Evaluación estratificada por clase",
                "Sistema de detección de sesgos automatizado"
            ]
        }
        
        # Implementation note.
        requires_retraining = {
            "🔄 REQUIEREN REENTRENAMIENTO COMPLETO": [
                "Diversificación geográfica del dataset (+ razas asiáticas/africanas)",
                "Balanceo de tamaños físicos (+ razas grandes)",
                "Data augmentation específica para clases problemáticas",
                "Arquitectura mejorada (ej. EfficientNet, Vision Transformer)",
                "Transfer learning con modelos más recientes",
                "Entrenamiento multi-tarea (detección + clasificación)"
            ],
            "🎯 REQUIEREN FINE-TUNING DIRIGIDO": [
                "Reentrenamiento solo de clases problemáticas",
                "Ajuste de learning rate por clase",
                "Weighted loss para clases desbalanceadas",
                "Focal loss para clases difíciles",
                "Class-balanced sampling durante entrenamiento",
                "Mixup/CutMix específico para clases problemáticas"
            ]
        }
        
        # Mejoras of post-processing
        post_processing = {
            "⚡ POST-PROCESAMIENTO (SIN REENTRENAMIENTO)": [
                "Ensemble de múltiples modelos existentes",
                "Test-time augmentation (TTA)",
                "Calibración avanzada de probabilidades",
                "Filtros de confianza adaptativos",
                "Voting schemes para predicciones ambiguas",
                "Rejection sampling para predicciones inciertas"
            ]
        }
        
        all_strategies = {**no_retraining, **requires_retraining, **post_processing}
        
        for category, strategies in all_strategies.items():
            print(f"\n{category}:")
            for i, strategy in enumerate(strategies, 1):
                print(f"   {i}. {strategy}")
        
        return all_strategies
    
    def recommend_action_plan(self, performance_analysis):
        """Technical documentation in English."""
        print(f"\n🎯 RECOMENDACIÓN DE PLAN DE ACCIÓN")
        print("="*70)
        
        if not performance_analysis:
            print("❌ No se puede generar recomendación sin análisis de rendimiento")
            return None
        
        needs_retraining = performance_analysis['needs_retraining']
        priority = needs_retraining['priority']
        
        print(f"🚦 PRIORIDAD DE REENTRENAMIENTO: {priority}")
        
        if needs_retraining['recommended']:
            print(f"\n✅ SE RECOMIENDA REENTRENAMIENTO")
            print(f"📋 Razones:")
            for reason in needs_retraining['reasons']:
                print(f"   • {reason}")
        else:
            print(f"\n❌ NO SE REQUIERE REENTRENAMIENTO INMEDIATO")
            print(f"✅ Las mejoras implementadas son suficientes por ahora")
        
        # Implementation note.
        action_plan = self._create_specific_action_plan(performance_analysis)
        
        print(f"\n🚀 PLAN DE ACCIÓN RECOMENDADO:")
        for phase, actions in action_plan.items():
            print(f"\n📋 {phase}:")
            for i, action in enumerate(actions, 1):
                print(f"   {i}. {action}")
        
        return action_plan
    
    def _create_specific_action_plan(self, analysis):
        """Technical documentation in English."""
        needs_retraining = analysis['needs_retraining']
        problematic_count = len(analysis['problematic_classes'])
        performance_gap = analysis['performance_gap']
        
        if not needs_retraining['recommended']:
            return {
                "FASE 1 - OPTIMIZACIÓN ACTUAL (0-2 semanas)": [
                    "Optimizar umbrales adaptativos con más datos de validación",
                    "Implementar ensemble del modelo actual con diferentes temperaturas",
                    "Aplicar test-time augmentation para mejorar predicciones",
                    "Monitorear rendimiento con métricas por clase detalladas"
                ],
                "FASE 2 - EVALUACIÓN CONTINUA": [
                    "Recolectar feedback de usuarios reales",
                    "Analizar casos de fallo específicos", 
                    "Revisar necesidad de reentrenamiento en 3 meses"
                ]
            }
        elif needs_retraining['priority'] == 'MEDIUM':
            return {
                "FASE 1 - MEJORAS SIN REENTRENAMIENTO (1-2 semanas)": [
                    "Implementar ensemble de modelos existentes",
                    "Aplicar técnicas avanzadas de calibración",
                    "Test-time augmentation para clases problemáticas",
                    "Optimización de hiperparámetros de inferencia"
                ],
                "FASE 2 - FINE-TUNING DIRIGIDO (2-3 semanas)": [
                    f"Fine-tuning solo para las {problematic_count} clases más problemáticas",
                    "Aplicar weighted loss específico para clases difíciles",
                    "Data augmentation intensiva para clases problemáticas",
                    "Validación cruzada para evaluar mejoras"
                ]
            }
        else:  # HIGH priority
            return {
                "FASE 1 - MEJORAS INMEDIATAS (1 semana)": [
                    "Implementar ensemble y TTA para mitigar problemas actuales",
                    "Aplicar rejection sampling para predicciones de baja confianza",
                    "Documentar limitaciones actuales para usuarios"
                ],
                "FASE 2 - REENTRENAMIENTO COMPLETO (3-4 semanas)": [
                    "Recopilar datos adicionales para clases problemáticas",
                    "Diversificar dataset geográficamente",
                    "Entrenar con arquitectura mejorada (EfficientNet-B4 o ViT)",
                    "Implementar técnicas de balanceo avanzadas"
                ],
                "FASE 3 - VALIDACIÓN Y DESPLIEGUE (1-2 semanas)": [
                    "Evaluación exhaustiva del nuevo modelo",
                    "A/B testing contra modelo actual",
                    "Despliegue gradual y monitoreo de rendimiento"
                ]
            }
    
    def estimate_improvement_potential(self, action_plan):
        """Estima the potencial of improvement of cada strategy"""
        print(f"\n📈 ESTIMACIÓN DE POTENCIAL DE MEJORA")
        print("="*70)
        
        current_performance = self.class_evaluation.get('overall_accuracy', 0.868)
        
        improvement_estimates = {
            "Optimización actual (sin reentrenamiento)": {
                "accuracy_gain": 0.02,  # +2%
                "time_investment": "1-2 semanas",
                "cost": "Bajo",
                "probability_success": 0.9
            },
            "Fine-tuning dirigido": {
                "accuracy_gain": 0.05,  # +5%
                "time_investment": "2-3 semanas", 
                "cost": "Medio",
                "probability_success": 0.7
            },
            "Reentrenamiento completo": {
                "accuracy_gain": 0.08,  # +8%
                "time_investment": "4-6 semanas",
                "cost": "Alto", 
                "probability_success": 0.6
            }
        }
        
        print(f"📊 RENDIMIENTO ACTUAL: {current_performance:.3f}")
        print(f"\n🎯 ESTIMACIONES DE MEJORA:")
        
        for strategy, estimates in improvement_estimates.items():
            projected_acc = current_performance + estimates['accuracy_gain']
            expected_gain = estimates['accuracy_gain'] * estimates['probability_success']
            
            print(f"\n📋 {strategy}:")
            print(f"   🎯 Ganancia estimada: +{estimates['accuracy_gain']:.3f} ({estimates['accuracy_gain']*100:.1f}%)")
            print(f"   📈 Accuracy proyectada: {projected_acc:.3f}")
            print(f"   📊 Ganancia esperada: +{expected_gain:.3f} ({expected_gain*100:.1f}%)")
            print(f"   ⏰ Tiempo: {estimates['time_investment']}")
            print(f"   💰 Costo: {estimates['cost']}")
            print(f"   📈 Probabilidad éxito: {estimates['probability_success']*100:.0f}%")
        
        return improvement_estimates
    
    def create_decision_matrix(self, performance_analysis, improvement_estimates):
        """Technical documentation in English."""
        print(f"\n📊 MATRIZ DE DECISIÓN")
        print("="*70)
        
        # Implementation note.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Implementation note.
        strategies = list(improvement_estimates.keys())
        gains = [est['accuracy_gain'] * est['probability_success'] for est in improvement_estimates.values()]
        times = [1.5, 2.5, 5.0]  # semanas average
        costs = ['Bajo', 'Medio', 'Alto']
        colors = ['green', 'orange', 'red']
        
        scatter = ax1.scatter(times, gains, s=200, c=colors, alpha=0.7)
        ax1.set_xlabel('Tiempo de Implementación (semanas)')
        ax1.set_ylabel('Ganancia Esperada de Accuracy')
        ax1.set_title('Ganancia Esperada vs Tiempo de Implementación')
        ax1.grid(True, alpha=0.3)
        
        # Add labels
        for i, (strategy, gain, time) in enumerate(zip(strategies, gains, times)):
            ax1.annotate(strategy.split('(')[0], (time, gain), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Implementation note.
        if self.class_evaluation and 'class_details' in self.class_evaluation:
            class_details = self.class_evaluation['class_details']
            accuracies = [details['accuracy'] for details in class_details.values()]
            
            ax2.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='navy')
            ax2.axvline(np.mean(accuracies), color='red', linestyle='--', 
                       label=f'Media: {np.mean(accuracies):.3f}')
            ax2.axvline(0.7, color='orange', linestyle='--', 
                       label='Umbral problemático')
            ax2.set_xlabel('Accuracy por Clase')
            ax2.set_ylabel('Número de Clases')
            ax2.set_title('Distribución de Accuracy por Clase')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig('retraining_decision_matrix.png', dpi=300, bbox_inches='tight')
        print("   ✅ Matriz de decisión guardada: retraining_decision_matrix.png")
        
        # Implementation note.
        needs_retraining = performance_analysis['needs_retraining']
        
        if needs_retraining['priority'] == 'LOW':
            recommendation = "Optimización actual"
            rationale = "Las mejoras ya implementadas son suficientes. Optimizar sin reentrenamiento."
        elif needs_retraining['priority'] == 'MEDIUM':
            recommendation = "Fine-tuning dirigido" 
            rationale = "Balance óptimo entre mejora esperada y esfuerzo requerido."
        else:
            recommendation = "Reentrenamiento completo"
            rationale = "Los problemas actuales requieren intervención fundamental."
        
        print(f"\n🎯 RECOMENDACIÓN FINAL: {recommendation}")
        print(f"💡 Justificación: {rationale}")
        
        return {
            'recommended_strategy': recommendation,
            'rationale': rationale,
            'visualization_path': 'retraining_decision_matrix.png'
        }
    
    def run_complete_analysis(self):
        """Technical documentation in English."""
        print("🔬" * 70)
        print("🔬 ANÁLISIS COMPLETO DE NECESIDAD DE REENTRENAMIENTO")
        print("🔬" * 70)
        
        # 1. Analizar performance actual
        performance_analysis = self.analyze_current_performance_gaps()
        
        # 2. Categorizar estrategias
        strategies = self.categorize_improvement_strategies()
        
        # Implementation note.
        action_plan = self.recommend_action_plan(performance_analysis)
        
        # 4. Estimar potencial of improvement
        improvement_estimates = self.estimate_improvement_potential(action_plan)
        
        # Implementation note.
        decision_matrix = self.create_decision_matrix(performance_analysis, improvement_estimates)
        
        # 6. Save reporte complete
        complete_report = {
            'timestamp': np.datetime64('now').item().isoformat(),
            'performance_analysis': performance_analysis,
            'improvement_strategies': strategies,
            'action_plan': action_plan,
            'improvement_estimates': improvement_estimates,
            'decision_matrix': decision_matrix
        }
        
        with open('retraining_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(complete_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ ANÁLISIS COMPLETO FINALIZADO")
        print(f"   📊 Reporte guardado: retraining_analysis_report.json")
        print(f"   📈 Visualización: retraining_decision_matrix.png")
        
        return complete_report

def main():
    """Function main"""
    workspace_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
    
    analyzer = RetrainingAnalyzer(workspace_path)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()