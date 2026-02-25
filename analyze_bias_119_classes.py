#!/usr/bin/env python3
"""
🔍 ANÁLISIS DE SESGO PARA MODELO DE 119 CLASES
============================================
Identificar razas con mayor posibilidad de sesgo basado en:
- Métricas de rendimiento por clase
- Similitudes visuales entre razas
- Distribución geográfica
- Tamaños y características físicas
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

class BiasAnalyzer119:
    def __init__(self):
        self.class_metrics = {}
        self.breed_names = []
        self.load_data()
        
    def load_data(self):
        """Cargar métricas de las 119 clases"""
        try:
            # Cargar métricas por clase
            with open('class_metrics.json', 'r') as f:
                self.class_metrics = json.load(f)
            
            # Obtener nombres de razas del modelo balanceado
            from balanced_model_server import CLASS_NAMES
            self.breed_names = [name.split('-')[1] if '-' in name else name for name in CLASS_NAMES]
            
            print(f"✅ Cargadas métricas de {len(self.class_metrics)} clases")
            print(f"✅ Nombres de {len(self.breed_names)} razas del modelo")
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
    
    def analyze_performance_bias(self):
        """Analizar sesgo basado en rendimiento por clase"""
        print("\n" + "="*60)
        print("📊 ANÁLISIS DE SESGO POR RENDIMIENTO")
        print("="*60)
        
        # Crear DataFrame con métricas
        df_data = []
        for breed, metrics in self.class_metrics.items():
            df_data.append({
                'breed': breed,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'accuracy': metrics.get('accuracy', 0),
                'avg_confidence': metrics.get('avg_confidence', 0),
                'std_confidence': metrics.get('std_confidence', 0),
                'support': metrics.get('support', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Identificar razas con peor rendimiento
        print("\n🔴 RAZAS CON MAYOR SESGO (Peor Rendimiento):")
        print("-" * 50)
        
        # Top 10 peores en F1-Score
        worst_f1 = df.nsmallest(10, 'f1_score')
        print("\n📉 Top 10 Peores F1-Score:")
        for idx, row in worst_f1.iterrows():
            print(f"  {row['breed']:25} | F1: {row['f1_score']:.3f} | Acc: {row['accuracy']:.3f}")
        
        # Top 10 peores en Recall (más falsos negativos)
        worst_recall = df.nsmallest(10, 'recall')
        print("\n⚠️  Top 10 Peores Recall (Más Falsos Negativos):")
        for idx, row in worst_recall.iterrows():
            print(f"  {row['breed']:25} | Recall: {row['recall']:.3f} | Precision: {row['precision']:.3f}")
        
        # Razas con alta variabilidad en confianza
        high_variance = df.nlargest(10, 'std_confidence')
        print("\n🌀 Top 10 Mayor Variabilidad en Confianza:")
        for idx, row in high_variance.iterrows():
            print(f"  {row['breed']:25} | Std: {row['std_confidence']:.3f} | Avg: {row['avg_confidence']:.3f}")
        
        return df
    
    def analyze_visual_similarity_bias(self):
        """Identificar grupos de razas visualmente similares propensas a confusión"""
        print("\n" + "="*60)
        print("👁️  ANÁLISIS DE SESGO POR SIMILITUD VISUAL")
        print("="*60)
        
        # Grupos de razas similares que pueden causar confusión
        similar_groups = {
            "Terriers Pequeños": [
                "Yorkshire_terrier", "cairn", "Norfolk_terrier", "Norwich_terrier",
                "West_Highland_white_terrier", "Scottish_terrier", "Australian_terrier"
            ],
            "Spaniels": [
                "Japanese_spaniel", "Blenheim_spaniel", "cocker_spaniel", 
                "English_springer", "Welsh_springer_spaniel", "Sussex_spaniel"
            ],
            "Pastores/Collies": [
                "collie", "Border_collie", "Shetland_sheepdog", "Old_English_sheepdog",
                "German_shepherd", "malinois", "groenendael"
            ],
            "Perros Nórdicos": [
                "Siberian_husky", "malamute", "Samoyed", "Eskimo_dog",
                "Norwegian_elkhound", "Pomeranian"
            ],
            "Galgos/Lebreles": [
                "Afghan_hound", "borzoi", "Italian_greyhound", "Ibizan_hound",
                "Saluki", "Scottish_deerhound", "whippet"
            ],
            "Bulldogs/Mastines": [
                "French_bulldog", "Boston_bull", "bull_mastiff", "Great_Dane",
                "Saint_Bernard", "Tibetan_mastiff"
            ],
            "Retrievers": [
                "golden_retriever", "Labrador_retriever", "flat-coated_retriever",
                "curly-coated_retriever", "Chesapeake_Bay_retriever"
            ],
            "Schnauzers": [
                "miniature_schnauzer", "giant_schnauzer", "standard_schnauzer"
            ]
        }
        
        bias_risk = {}
        
        for group_name, breeds in similar_groups.items():
            print(f"\n🔍 Grupo: {group_name}")
            group_metrics = []
            available_breeds = []
            
            for breed in breeds:
                if breed in self.class_metrics:
                    metrics = self.class_metrics[breed]
                    group_metrics.append({
                        'breed': breed,
                        'f1_score': metrics.get('f1_score', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0)
                    })
                    available_breeds.append(breed)
            
            if group_metrics:
                # Calcular varianza del grupo
                f1_scores = [m['f1_score'] for m in group_metrics]
                f1_variance = np.var(f1_scores)
                f1_mean = np.mean(f1_scores)
                
                bias_risk[group_name] = {
                    'variance': f1_variance,
                    'mean_f1': f1_mean,
                    'breeds': available_breeds,
                    'risk_level': 'ALTO' if f1_variance > 0.05 else 'MEDIO' if f1_variance > 0.02 else 'BAJO'
                }
                
                print(f"  📊 F1 Promedio: {f1_mean:.3f}")
                print(f"  🌀 Varianza F1: {f1_variance:.4f}")
                print(f"  ⚠️  Riesgo de Sesgo: {bias_risk[group_name]['risk_level']}")
                
                # Mostrar peores del grupo
                worst_in_group = sorted(group_metrics, key=lambda x: x['f1_score'])[:3]
                print(f"  🔴 Peores del grupo:")
                for breed_data in worst_in_group:
                    print(f"    - {breed_data['breed']:20} F1: {breed_data['f1_score']:.3f}")
        
        return bias_risk
    
    def analyze_geographic_bias(self):
        """Analizar sesgo geográfico basado en origen de las razas"""
        print("\n" + "="*60)
        print("🌍 ANÁLISIS DE SESGO GEOGRÁFICO")
        print("="*60)
        
        # Clasificación geográfica aproximada de razas
        geographic_regions = {
            "Europa Occidental": [
                "German_shepherd", "Rottweiler", "Doberman", "Great_Dane", "boxer",
                "German_short-haired_pointer", "Weimaraner", "giant_schnauzer",
                "standard_schnauzer", "miniature_schnauzer", "Bernese_mountain_dog"
            ],
            "Reino Unido": [
                "English_foxhound", "English_setter", "English_springer", "cocker_spaniel",
                "Yorkshire_terrier", "West_Highland_white_terrier", "Scottish_terrier",
                "Border_collie", "collie", "Shetland_sheepdog", "cairn", "Norfolk_terrier",
                "Norwich_terrier", "Airedale", "Border_terrier", "Bedlington_terrier"
            ],
            "Francia": [
                "Brittany_spaniel", "papillon", "Bouvier_des_Flandres", "briard",
                "French_bulldog"
            ],
            "Escandinavia": [
                "Norwegian_elkhound", "Siberian_husky", "malamute", "Samoyed",
                "Eskimo_dog"
            ],
            "Asia": [
                "chow", "Pomeranian", "Japanese_spaniel", "Shih-Tzu", "Lhasa",
                "Tibetan_terrier", "Tibetan_mastiff", "basenji"  # basenji es africano
            ],
            "Mediterráneo": [
                "Italian_greyhound", "Ibizan_hound", "Saluki"
            ],
            "América": [
                "American_Staffordshire_terrier", "Boston_bull", "Chesapeake_Bay_retriever"
            ]
        }
        
        regional_performance = {}
        
        for region, breeds in geographic_regions.items():
            f1_scores = []
            available_breeds = []
            
            for breed in breeds:
                if breed in self.class_metrics:
                    f1_scores.append(self.class_metrics[breed].get('f1_score', 0))
                    available_breeds.append(breed)
            
            if f1_scores:
                regional_performance[region] = {
                    'mean_f1': np.mean(f1_scores),
                    'std_f1': np.std(f1_scores),
                    'count': len(f1_scores),
                    'breeds': available_breeds
                }
        
        print("\n📊 Rendimiento por Región:")
        print("-" * 40)
        sorted_regions = sorted(regional_performance.items(), 
                              key=lambda x: x[1]['mean_f1'], reverse=True)
        
        for region, data in sorted_regions:
            print(f"{region:20} | F1: {data['mean_f1']:.3f} ± {data['std_f1']:.3f} | Razas: {data['count']}")
        
        # Identificar regiones con sesgo
        all_f1_means = [data['mean_f1'] for data in regional_performance.values()]
        global_mean = np.mean(all_f1_means)
        
        print(f"\n🎯 F1 Global Promedio: {global_mean:.3f}")
        print("\n⚠️  Regiones con Posible Sesgo:")
        print("-" * 40)
        
        for region, data in sorted_regions:
            if data['mean_f1'] < global_mean - 0.05:
                print(f"🔴 {region}: {data['mean_f1']:.3f} (BAJO RENDIMIENTO)")
            elif data['mean_f1'] > global_mean + 0.05:
                print(f"🟢 {region}: {data['mean_f1']:.3f} (SOBRERREPRESENTADO)")
        
        return regional_performance
    
    def generate_bias_report(self):
        """Generar reporte completo de sesgo"""
        print("\n" + "="*70)
        print("📋 REPORTE COMPLETO DE ANÁLISIS DE SESGO - 119 CLASES")
        print("="*70)
        
        # Ejecutar todos los análisis
        df_performance = self.analyze_performance_bias()
        visual_bias = self.analyze_visual_similarity_bias()
        geographic_bias = self.analyze_geographic_bias()
        
        # Generar recomendaciones específicas
        print("\n" + "="*60)
        print("🎯 RAZAS CON MAYOR RIESGO DE SESGO")
        print("="*60)
        
        # Combinar todos los análisis para ranking final
        high_risk_breeds = set()
        
        # De rendimiento (peores 15)
        worst_performers = df_performance.nsmallest(15, 'f1_score')['breed'].tolist()
        high_risk_breeds.update(worst_performers)
        
        # De similitud visual (grupos de alto riesgo)
        for group, data in visual_bias.items():
            if data['risk_level'] == 'ALTO':
                high_risk_breeds.update(data['breeds'])
        
        print(f"\n🚨 TOP RAZAS DE ALTO RIESGO ({len(high_risk_breeds)} total):")
        print("-" * 50)
        
        for i, breed in enumerate(sorted(high_risk_breeds), 1):
            if breed in self.class_metrics:
                metrics = self.class_metrics[breed]
                f1 = metrics.get('f1_score', 0)
                recall = metrics.get('recall', 0)
                precision = metrics.get('precision', 0)
                print(f"{i:2}. {breed:25} | F1: {f1:.3f} | P: {precision:.3f} | R: {recall:.3f}")
        
        # Recomendaciones específicas
        print("\n" + "="*60)
        print("💡 RECOMENDACIONES ESPECÍFICAS")
        print("="*60)
        
        recommendations = [
            "1. 🎯 ENFOQUE PRIORITARIO en razas con F1 < 0.70",
            "2. 🔄 AUMENTAR datos de entrenamiento para razas de bajo rendimiento",
            "3. 👁️  TÉCNICAS DE DIFERENCIACIÓN para grupos visualmente similares",
            "4. 🌍 BALANCEAR representación geográfica en el dataset",
            "5. 🧠 FINE-TUNING específico para razas problemáticas",
            "6. 📊 UMBRALES ADAPTATIVOS por raza según rendimiento histórico",
            "7. 🔍 AUGMENTACIÓN ESPECIALIZADA para razas confusas",
            "8. ⚖️  WEIGHTED LOSS por clase durante reentrenamiento"
        ]
        
        for rec in recommendations:
            print(rec)
        
        # Guardar reporte
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_classes': len(self.class_metrics),
            'high_risk_breeds': list(high_risk_breeds),
            'performance_metrics': df_performance.to_dict('records'),
            'visual_similarity_risk': visual_bias,
            'geographic_performance': geographic_bias,
            'recommendations': recommendations
        }
        
        with open('bias_analysis_119_classes.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Reporte guardado en: bias_analysis_119_classes.json")
        
        return report_data

def main():
    """Ejecutar análisis completo de sesgo"""
    print("🔍 Iniciando Análisis de Sesgo para Modelo de 119 Clases...")
    
    analyzer = BiasAnalyzer119()
    
    if not analyzer.class_metrics:
        print("❌ No se pudieron cargar las métricas. Verifica que class_metrics.json existe.")
        return
    
    # Generar reporte completo
    report = analyzer.generate_bias_report()
    
    print("\n✅ Análisis de sesgo completado exitosamente!")

if __name__ == "__main__":
    main()