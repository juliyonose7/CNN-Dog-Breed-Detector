# !/usr/bin/env python3
"""
Technical documentation in English.
==================================================
Identificar breeds with recall bajo that generan muchos false negatives
Technical documentation in English.
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

class FalseNegativeAnalyzer:
    def __init__(self):
        self.class_metrics = {}
        self.load_data()
        
    def load_data(self):
        """Technical documentation in English."""
        try:
            with open('class_metrics.json', 'r') as f:
                self.class_metrics = json.load(f)
            
            print(f"✅ Cargadas métricas de {len(self.class_metrics)} clases")
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
    
    def analyze_false_negatives(self):
        """Analizar breeds with alto riesgo of false negatives"""
        print("\n" + "="*70)
        print("🔴 ANÁLISIS DE FALSOS NEGATIVOS - RECALL BAJO")
        print("="*70)
        
        # Implementation note.
        df_data = []
        for breed, metrics in self.class_metrics.items():
            recall = metrics.get('recall', 0)
            precision = metrics.get('precision', 0)
            f1_score = metrics.get('f1_score', 0)
            support = metrics.get('support', 0)
            accuracy = metrics.get('accuracy', 0)
            avg_confidence = metrics.get('avg_confidence', 0)
            std_confidence = metrics.get('std_confidence', 0)
            
            # Calcular false negatives aproximados
            true_positives = recall * support
            false_negatives = support - true_positives
            false_negative_rate = false_negatives / support if support > 0 else 0
            
            df_data.append({
                'breed': breed,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score,
                'support': support,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'std_confidence': std_confidence,
                'false_negatives': false_negatives,
                'false_negative_rate': false_negative_rate
            })
        
        df = pd.DataFrame(df_data)
        
        # Ordenar for recall ascendente (peores primero)
        df_sorted = df.sort_values('recall')
        
        print("\n🚨 TOP 15 RAZAS CON MÁS FALSOS NEGATIVOS (Recall más bajo):")
        print("=" * 80)
        print(f"{'Raza':25} | {'Recall':6} | {'FN':3} | {'FN%':5} | {'Prec':6} | {'F1':6} | {'Conf':6}")
        print("=" * 80)
        
        worst_recall_breeds = []
        
        for idx, row in df_sorted.head(15).iterrows():
            breed = row['breed']
            recall = row['recall']
            fn_count = int(row['false_negatives'])
            fn_rate = row['false_negative_rate']
            precision = row['precision']
            f1 = row['f1_score']
            confidence = row['avg_confidence']
            
            # Clasificar severidad
            if recall < 0.50:
                severity = "🔴 CRÍTICO"
            elif recall < 0.70:
                severity = "🟠 ALTO"
            elif recall < 0.85:
                severity = "🟡 MEDIO"
            else:
                severity = "🟢 BAJO"
            
            print(f"{breed[:24]:25} | {recall:.3f} | {fn_count:3} | {fn_rate:.1%} | {precision:.3f} | {f1:.3f} | {confidence:.3f}")
            
            worst_recall_breeds.append({
                'breed': breed,
                'recall': recall,
                'false_negatives': fn_count,
                'severity': severity,
                'precision': precision,
                'f1_score': f1
            })
        
        return worst_recall_breeds, df
    
    def categorize_false_negative_causes(self, worst_breeds):
        """Categorizar the causas probables of the false negatives"""
        print("\n" + "="*70)
        print("🔍 ANÁLISIS DE CAUSAS DE FALSOS NEGATIVOS")
        print("="*70)
        
        # Implementation note.
        similar_groups = {
            "Terriers Pequeños": [
                "Norfolk_terrier", "Norwich_terrier", "cairn", "Yorkshire_terrier",
                "West_Highland_white_terrier", "Scottish_terrier", "Australian_terrier",
                "toy_terrier", "Lakeland_terrier", "Border_terrier"
            ],
            "Perros Nórdicos/Spitz": [
                "Siberian_husky", "malamute", "Samoyed", "Eskimo_dog",
                "Norwegian_elkhound", "Pomeranian", "keeshond", "chow"
            ],
            "Galgos/Lebreles": [
                "whippet", "Italian_greyhound", "Afghan_hound", "borzoi",
                "Ibizan_hound", "Saluki", "Scottish_deerhound"
            ],
            "Spaniels": [
                "cocker_spaniel", "English_springer", "Welsh_springer_spaniel",
                "Japanese_spaniel", "Blenheim_spaniel", "Sussex_spaniel"
            ],
            "Pastores": [
                "German_shepherd", "collie", "Border_collie", "Shetland_sheepdog",
                "Old_English_sheepdog", "malinois", "groenendael"
            ]
        }
        
        print("\n📊 CATEGORIZACIÓN POR GRUPOS PROBLEMÁTICOS:")
        print("-" * 50)
        
        group_problems = {}
        
        for group_name, breeds in similar_groups.items():
            group_false_negatives = []
            
            for breed_data in worst_breeds:
                breed = breed_data['breed']
                if breed in breeds:
                    group_false_negatives.append(breed_data)
            
            if group_false_negatives:
                group_problems[group_name] = group_false_negatives
                
                print(f"\n🔍 Grupo: {group_name}")
                print(f"   Razas problemáticas: {len(group_false_negatives)}")
                
                for breed_data in group_false_negatives:
                    breed = breed_data['breed']
                    recall = breed_data['recall']
                    fn_count = breed_data['false_negatives']
                    severity = breed_data['severity']
                    
                    print(f"   - {breed:20} | Recall: {recall:.3f} | FN: {fn_count:2} | {severity}")
        
        return group_problems
    
    def analyze_recall_vs_precision_balance(self, df):
        """Analizar the balance entre recall and precision"""
        print("\n" + "="*70)
        print("⚖️  ANÁLISIS DE BALANCE RECALL vs PRECISION")
        print("="*70)
        
        # Identificar casos where recall << precision (muchos false negatives)
        df['recall_precision_diff'] = df['precision'] - df['recall']
        
        # Casos where precision es mucho mayor that recall
        high_imbalance = df[df['recall_precision_diff'] > 0.2].sort_values('recall_precision_diff', ascending=False)
        
        print("\n🎯 RAZAS CON DESEQUILIBRIO RECALL << PRECISION:")
        print("   (Modelo muy conservador - genera muchos falsos negativos)")
        print("-" * 65)
        print(f"{'Raza':25} | {'Recall':6} | {'Prec':6} | {'Diff':6} | {'Interpretación'}")
        print("-" * 65)
        
        for idx, row in high_imbalance.head(10).iterrows():
            breed = row['breed']
            recall = row['recall']
            precision = row['precision']
            diff = row['recall_precision_diff']
            
            if diff > 0.4:
                interpretation = "MUY CONSERVADOR"
            elif diff > 0.3:
                interpretation = "CONSERVADOR"
            elif diff > 0.2:
                interpretation = "ALGO CONSERVADOR"
            else:
                interpretation = "BALANCEADO"
            
            print(f"{breed[:24]:25} | {recall:.3f} | {precision:.3f} | {diff:+.3f} | {interpretation}")
        
        return high_imbalance
    
    def generate_false_negative_recommendations(self, worst_breeds, group_problems, imbalanced_breeds):
        """Technical documentation in English."""
        print("\n" + "="*70)
        print("💡 RECOMENDACIONES PARA REDUCIR FALSOS NEGATIVOS")
        print("="*70)
        
        # Implementation note.
        critical_breeds = [b for b in worst_breeds if b['recall'] < 0.60]
        high_priority_breeds = [b for b in worst_breeds if 0.60 <= b['recall'] < 0.75]
        
        print(f"\n🚨 ATENCIÓN CRÍTICA ({len(critical_breeds)} razas con Recall < 0.60):")
        print("-" * 50)
        for breed_data in critical_breeds:
            breed = breed_data['breed']
            recall = breed_data['recall']
            fn_count = breed_data['false_negatives']
            print(f"  🔴 {breed:25} | Recall: {recall:.3f} | FN: {fn_count:2}")
        
        print(f"\n⚠️  ALTA PRIORIDAD ({len(high_priority_breeds)} razas con Recall 0.60-0.75):")
        print("-" * 50)
        for breed_data in high_priority_breeds:
            breed = breed_data['breed']
            recall = breed_data['recall']
            fn_count = breed_data['false_negatives']
            print(f"  🟠 {breed:25} | Recall: {recall:.3f} | FN: {fn_count:2}")
        
        print("\n🛠️  ESTRATEGIAS ESPECÍFICAS:")
        print("-" * 40)
        
        strategies = [
            "1. 📈 AUMENTAR THRESHOLD de clasificación para razas conservadoras",
            "2. 🎯 WEIGHTED LOSS function con penalización extra por falsos negativos",
            "3. 🔄 DATA AUGMENTATION específica para razas con pocos ejemplos detectados",
            "4. 🧠 FOCAL LOSS para balancear clases difíciles",
            "5. 📊 ENSEMBLE METHODS para mejorar sensibilidad",
            "6. 🎨 FEATURE ENHANCEMENT para características distintivas",
            "7. ⚖️  THRESHOLD TUNING por raza individual",
            "8. 🔍 HARD NEGATIVE MINING para casos difíciles"
        ]
        
        for strategy in strategies:
            print(f"  {strategy}")
        
        print("\n🎯 ACCIONES INMEDIATAS POR GRUPO:")
        print("-" * 35)
        
        if "Terriers Pequeños" in group_problems:
            print("  🐕 TERRIERS PEQUEÑOS:")
            print("     - Enfocarse en diferencias sutiles de orejas y pelaje")
            print("     - Augmentación con variaciones de ángulo y postura")
        
        if "Perros Nórdicos/Spitz" in group_problems:
            print("  ❄️  PERROS NÓRDICOS:")
            print("     - Destacar diferencias de tamaño y forma de cola")
            print("     - Más datos de diferentes estaciones/backgrounds")
        
        if "Galgos/Lebreles" in group_problems:
            print("  🏃 GALGOS/LEBRELES:")
            print("     - Enfoque en proporciones corporales específicas")
            print("     - Imágenes de cuerpo completo, no solo cabeza")
        
        # Save reporte
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'analysis_type': 'false_negatives',
            'total_breeds_analyzed': len(self.class_metrics),
            'critical_breeds': [b['breed'] for b in critical_breeds],
            'high_priority_breeds': [b['breed'] for b in high_priority_breeds],
            'worst_recall_breeds': worst_breeds,
            'group_problems': {k: [b['breed'] for b in v] for k, v in group_problems.items()},
            'recommendations': strategies
        }
        
        with open('false_negatives_analysis_119.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Reporte guardado en: false_negatives_analysis_119.json")
        
        return report_data

def main():
    """Technical documentation in English."""
    print("🔍 Iniciando Análisis de Falsos Negativos - Modelo 119 Clases...")
    
    analyzer = FalseNegativeAnalyzer()
    
    if not analyzer.class_metrics:
        print("❌ No se pudieron cargar las métricas. Verifica que class_metrics.json existe.")
        return
    
    # Implementation note.
    worst_breeds, df = analyzer.analyze_false_negatives()
    
    # Categorizar causas
    group_problems = analyzer.categorize_false_negative_causes(worst_breeds)
    
    # Analizar balance recall vs precision
    imbalanced_breeds = analyzer.analyze_recall_vs_precision_balance(df)
    
    # Generar recomendaciones
    report = analyzer.generate_false_negative_recommendations(worst_breeds, group_problems, imbalanced_breeds)
    
    print("\n✅ Análisis de falsos negativos completado!")
    print(f"📊 {len(worst_breeds)} razas identificadas con problemas de recall")
    print(f"🎯 {len([b for b in worst_breeds if b['recall'] < 0.60])} razas necesitan atención crítica")

if __name__ == "__main__":
    main()