#!/usr/bin/env python3
"""
📊 GENERADOR DE MÉTRICAS DETALLADAS POR CLASE
==============================================

Evalúa el rendimiento detallado del modelo unificado para cada una de las 50 razas:
- Precision, Recall, F1-Score por raza
- Matriz de confusión
- Análisis de errores comunes
- Identificación de razas problemáticas

Autor: Sistema IA  
Fecha: 2024
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class DetailedClassEvaluator:
    def __init__(self, model_path="balanced_models/best_balanced_breed_model_epoch_20_acc_88.1366.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar modelo
        self.model = None
        self.breed_classes = []
        self._load_model_and_classes()
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model_and_classes(self):
        """Carga el modelo y las clases"""
        print("📁 Cargando modelo para evaluación detallada...")
        
        # Definir arquitectura del modelo balanceado
        from torch import nn
        from torchvision import models
        
        class BalancedBreedClassifier(nn.Module):
            def __init__(self, num_classes=50):
                super().__init__()
                self.backbone = models.resnet50(weights=None)
                num_ftrs = self.backbone.fc.in_features
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_ftrs, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
            def forward(self, x):
                return self.backbone(x)
        
        if os.path.exists(self.model_path):
            self.model = BalancedBreedClassifier(num_classes=50).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ Modelo cargado: {checkpoint.get('val_accuracy', 0):.2f}% accuracy")
        else:
            print(f"❌ Modelo no encontrado: {self.model_path}")
            return
        
        # Cargar clases
        breed_data_path = "breed_processed_data/train"
        if os.path.exists(breed_data_path):
            self.breed_classes = sorted([d for d in os.listdir(breed_data_path) 
                                       if os.path.isdir(os.path.join(breed_data_path, d))])
            print(f"📋 Cargadas {len(self.breed_classes)} clases")
        else:
            print("❌ No se encontró directorio de clases")
    
    def evaluate_all_classes(self, test_data_path="breed_processed_data/val", samples_per_class=100):
        """Evalúa todas las clases con métricas detalladas"""
        print(f"📊 EVALUANDO {len(self.breed_classes)} CLASES...")
        print("="*60)
        
        if not os.path.exists(test_data_path):
            print(f"❌ Directorio de prueba no encontrado: {test_data_path}")
            return None
        
        # Recopilar predicciones y etiquetas reales
        all_true_labels = []
        all_predicted_labels = []
        all_probabilities = []
        class_details = {}
        
        for class_idx, breed_name in enumerate(self.breed_classes):
            print(f"🔍 Evaluando {breed_name} ({class_idx+1}/{len(self.breed_classes)})...")
            
            breed_path = os.path.join(test_data_path, breed_name)
            if not os.path.exists(breed_path):
                print(f"   ⚠️ Directorio no encontrado: {breed_path}")
                continue
            
            # Obtener imágenes de la clase
            image_files = [f for f in os.listdir(breed_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(image_files) == 0:
                print(f"   ⚠️ No se encontraron imágenes para {breed_name}")
                continue
            
            # Limitar número de muestras para rapidez
            sample_files = image_files[:min(samples_per_class, len(image_files))]
            
            breed_true_labels = []
            breed_predicted_labels = []
            breed_probabilities = []
            breed_confidences = []
            correct_predictions = 0
            
            for image_file in sample_files:
                try:
                    image_path = os.path.join(breed_path, image_file)
                    image = Image.open(image_path).convert('RGB')
                    input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probabilities = F.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0, predicted_class].item()
                        
                        breed_true_labels.append(class_idx)
                        breed_predicted_labels.append(predicted_class)
                        breed_probabilities.append(probabilities[0].cpu().numpy())
                        breed_confidences.append(confidence)
                        
                        if predicted_class == class_idx:
                            correct_predictions += 1
                        
                except Exception as e:
                    print(f"   ⚠️ Error con {image_file}: {e}")
                    continue
            
            # Calcular métricas para esta clase
            if len(breed_true_labels) > 0:
                breed_accuracy = correct_predictions / len(breed_true_labels)
                avg_confidence = np.mean(breed_confidences)
                std_confidence = np.std(breed_confidences)
                
                class_details[breed_name] = {
                    'class_index': class_idx,
                    'samples_evaluated': len(breed_true_labels),
                    'correct_predictions': correct_predictions,
                    'accuracy': breed_accuracy,
                    'avg_confidence': avg_confidence,
                    'std_confidence': std_confidence,
                    'min_confidence': min(breed_confidences),
                    'max_confidence': max(breed_confidences)
                }
                
                print(f"   ✅ Accuracy: {breed_accuracy:.3f} | "
                      f"Confianza: {avg_confidence:.3f}±{std_confidence:.3f} | "
                      f"Muestras: {len(breed_true_labels)}")
                
                # Agregar a listas globales
                all_true_labels.extend(breed_true_labels)
                all_predicted_labels.extend(breed_predicted_labels)
                all_probabilities.extend(breed_probabilities)
        
        # Calcular métricas globales
        print(f"\n📊 CALCULANDO MÉTRICAS GLOBALES...")
        
        # Classification report detallado
        class_names = [self.breed_classes[i] for i in range(len(self.breed_classes))]
        report = classification_report(all_true_labels, all_predicted_labels, 
                                     target_names=class_names, output_dict=True)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        
        # Resultados finales
        evaluation_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_path': self.model_path,
            'samples_per_class': samples_per_class,
            'total_samples': len(all_true_labels),
            'overall_accuracy': report['accuracy'],
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg'],
            'class_details': class_details,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Identificar clases problemáticas
        problematic_classes = []
        excellent_classes = []
        
        for breed_name, details in class_details.items():
            if details['accuracy'] < 0.7:
                problematic_classes.append((breed_name, details['accuracy']))
            elif details['accuracy'] > 0.95:
                excellent_classes.append((breed_name, details['accuracy']))
        
        problematic_classes.sort(key=lambda x: x[1])  # Ordenar por accuracy ascendente
        excellent_classes.sort(key=lambda x: x[1], reverse=True)  # Descendente
        
        evaluation_results['problematic_classes'] = problematic_classes
        evaluation_results['excellent_classes'] = excellent_classes
        
        # Mostrar resumen
        print(f"\n📈 RESUMEN DE EVALUACIÓN:")
        print(f"   Overall Accuracy: {report['accuracy']:.4f}")
        print(f"   Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"   Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
        print(f"   Total muestras: {len(all_true_labels):,}")
        
        print(f"\n🚨 CLASES PROBLEMÁTICAS (accuracy < 0.70):")
        for breed, acc in problematic_classes[:10]:
            print(f"   {breed}: {acc:.3f}")
        
        print(f"\n✅ CLASES EXCELENTES (accuracy > 0.95):")
        for breed, acc in excellent_classes[:10]:
            print(f"   {breed}: {acc:.3f}")
        
        return evaluation_results
    
    def calculate_per_class_metrics(self, evaluation_results):
        """Calcula métricas precisas por clase individual"""
        print(f"\n🎯 CALCULANDO MÉTRICAS PRECISAS POR CLASE...")
        
        if not evaluation_results:
            return None
        
        per_class_metrics = {}
        report = evaluation_results['classification_report']
        
        for breed_name in self.breed_classes:
            if breed_name in report:
                breed_report = report[breed_name]
                class_details = evaluation_results['class_details'].get(breed_name, {})
                
                per_class_metrics[breed_name] = {
                    'precision': breed_report['precision'],
                    'recall': breed_report['recall'],
                    'f1_score': breed_report['f1-score'],
                    'support': breed_report['support'],
                    'accuracy': class_details.get('accuracy', 0.0),
                    'avg_confidence': class_details.get('avg_confidence', 0.0),
                    'std_confidence': class_details.get('std_confidence', 0.0),
                    'samples_evaluated': class_details.get('samples_evaluated', 0)
                }
        
        # Guardar métricas por clase
        with open('class_metrics.json', 'w') as f:
            json.dump(per_class_metrics, f, indent=2, default=str)
        
        print(f"✅ Métricas por clase guardadas: class_metrics.json")
        return per_class_metrics
    
    def compute_optimal_thresholds(self, evaluation_results):
        """Calcula umbrales óptimos por clase usando ROC"""
        print(f"\n🎯 CALCULANDO UMBRALES ÓPTIMOS POR CLASE...")
        
        # Este sería un cálculo más avanzado que requeriría 
        # predicciones probabilísticas por clase
        # Por simplicidad, usaremos heurísticas basadas en el rendimiento
        
        optimal_thresholds = {}
        
        if evaluation_results and 'class_details' in evaluation_results:
            for breed_name, details in evaluation_results['class_details'].items():
                accuracy = details['accuracy']
                avg_confidence = details['avg_confidence']
                std_confidence = details['std_confidence']
                
                # Heurística: ajustar umbral basado en rendimiento
                if accuracy > 0.9:
                    # Clase con buen rendimiento: umbral más permisivo
                    threshold = max(0.2, avg_confidence - std_confidence)
                elif accuracy > 0.7:
                    # Clase con rendimiento moderado: umbral estándar
                    threshold = max(0.3, avg_confidence - 0.5 * std_confidence)
                else:
                    # Clase problemática: umbral más estricto
                    threshold = max(0.4, avg_confidence)
                
                optimal_thresholds[breed_name] = min(0.8, threshold)
        
        # Guardar umbrales
        with open('adaptive_thresholds.json', 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        
        print(f"✅ Umbrales adaptativos calculados para {len(optimal_thresholds)} clases")
        print(f"   Rango: {min(optimal_thresholds.values()):.3f} - {max(optimal_thresholds.values()):.3f}")
        print(f"✅ Guardados: adaptive_thresholds.json")
        
        return optimal_thresholds
    
    def create_detailed_visualizations(self, evaluation_results):
        """Crea visualizaciones detalladas de las métricas"""
        print(f"\n📊 CREANDO VISUALIZACIONES DETALLADAS...")
        
        if not evaluation_results:
            return None
        
        # Configurar matplotlib
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # 1. Accuracy por clase
        class_details = evaluation_results['class_details']
        breeds = list(class_details.keys())
        accuracies = [class_details[breed]['accuracy'] for breed in breeds]
        
        ax = axes[0]
        bars = ax.bar(range(len(breeds)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_title('Accuracy por Clase', fontweight='bold', fontsize=14)
        ax.set_xlabel('Razas')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(0, len(breeds), max(1, len(breeds)//10)))
        ax.set_xticklabels([breeds[i] for i in range(0, len(breeds), max(1, len(breeds)//10))], rotation=45, ha='right')
        ax.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Media: {np.mean(accuracies):.3f}')
        ax.legend()
        
        # 2. Distribución de accuracies
        ax = axes[1]
        ax.hist(accuracies, bins=15, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax.set_title('Distribución de Accuracies', fontweight='bold', fontsize=14)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Número de Clases')
        ax.axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Media: {np.mean(accuracies):.3f}')
        ax.legend()
        
        # 3. Confianza promedio vs Accuracy
        confidences = [class_details[breed]['avg_confidence'] for breed in breeds]
        
        ax = axes[2]
        scatter = ax.scatter(confidences, accuracies, c=accuracies, cmap='RdYlGn', s=50, alpha=0.7)
        ax.set_title('Confianza vs Accuracy por Clase', fontweight='bold', fontsize=14)
        ax.set_xlabel('Confianza Promedio')
        ax.set_ylabel('Accuracy')
        plt.colorbar(scatter, ax=ax, label='Accuracy')
        
        # Línea de tendencia
        z = np.polyfit(confidences, accuracies, 1)
        p = np.poly1d(z)
        ax.plot(confidences, p(confidences), "r--", alpha=0.8, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()
        
        # 4. Top 10 mejores y peores clases
        sorted_by_accuracy = sorted(class_details.items(), key=lambda x: x[1]['accuracy'])
        worst_10 = sorted_by_accuracy[:10]
        best_10 = sorted_by_accuracy[-10:]
        
        ax = axes[3]
        worst_names = [item[0][:15] for item in worst_10]  # Truncar nombres
        worst_accs = [item[1]['accuracy'] for item in worst_10]
        bars = ax.barh(range(len(worst_names)), worst_accs, color='lightcoral', edgecolor='darkred')
        ax.set_title('Top 10 Clases Más Problemáticas', fontweight='bold', fontsize=14)
        ax.set_xlabel('Accuracy')
        ax.set_yticks(range(len(worst_names)))
        ax.set_yticklabels(worst_names, fontsize=10)
        
        # Agregar valores
        for i, (bar, acc) in enumerate(zip(bars, worst_accs)):
            ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=9)
        
        # 5. Matriz de confusión (subset)
        conf_matrix = np.array(evaluation_results['confusion_matrix'])
        
        # Mostrar solo las primeras 20x20 para legibilidad
        subset_size = min(20, len(self.breed_classes))
        conf_subset = conf_matrix[:subset_size, :subset_size]
        
        ax = axes[4]
        im = ax.imshow(conf_subset, cmap='Blues', interpolation='nearest')
        ax.set_title(f'Matriz de Confusión (Primeras {subset_size} clases)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Realidad')
        
        # Colorbar
        plt.colorbar(im, ax=ax)
        
        # 6. Métricas resumen
        ax = axes[5]
        ax.axis('off')
        
        # Estadísticas resumen
        overall_acc = evaluation_results['overall_accuracy']
        macro_f1 = evaluation_results['macro_avg']['f1-score']
        weighted_f1 = evaluation_results['weighted_avg']['f1-score']
        
        problematic_count = len([acc for acc in accuracies if acc < 0.7])
        excellent_count = len([acc for acc in accuracies if acc > 0.9])
        
        summary_text = f"""
📊 RESUMEN DE MÉTRICAS DETALLADAS

🎯 Overall Accuracy: {overall_acc:.4f}
📈 Macro Avg F1: {macro_f1:.4f}
⚖️ Weighted Avg F1: {weighted_f1:.4f}

📋 Total de clases: {len(breeds)}
🚨 Clases problemáticas (<0.70): {problematic_count}
✅ Clases excelentes (>0.90): {excellent_count}
📊 Clases intermedias: {len(breeds) - problematic_count - excellent_count}

📈 Accuracy promedio: {np.mean(accuracies):.3f}
📊 Desviación estándar: {np.std(accuracies):.3f}
📉 Accuracy mínima: {min(accuracies):.3f}
📈 Accuracy máxima: {max(accuracies):.3f}

🔧 Modelo: ResNet50 Balanceado
✅ Arquitectura unificada (sin sesgos)
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('detailed_class_evaluation_report.png', dpi=300, bbox_inches='tight')
        print("   ✅ Visualización guardada: detailed_class_evaluation_report.png")
        
        return fig
    
    def generate_complete_report(self, samples_per_class=50):
        """Genera reporte completo de evaluación por clases"""
        print("📊" * 60)
        print("📊 GENERANDO REPORTE COMPLETO DE MÉTRICAS POR CLASE")
        print("📊" * 60)
        
        if self.model is None:
            print("❌ Modelo no cargado correctamente")
            return None
        
        # 1. Evaluar todas las clases
        evaluation_results = self.evaluate_all_classes(samples_per_class=samples_per_class)
        
        if not evaluation_results:
            print("❌ Error en la evaluación")
            return None
        
        # 2. Calcular métricas precisas por clase
        per_class_metrics = self.calculate_per_class_metrics(evaluation_results)
        
        # 3. Calcular umbrales óptimos
        optimal_thresholds = self.compute_optimal_thresholds(evaluation_results)
        
        # 4. Crear visualizaciones
        fig = self.create_detailed_visualizations(evaluation_results)
        
        # 5. Guardar reporte completo
        complete_report = {
            **evaluation_results,
            'per_class_metrics': per_class_metrics,
            'optimal_thresholds': optimal_thresholds
        }
        
        with open('complete_class_evaluation_report.json', 'w') as f:
            json.dump(complete_report, f, indent=2, default=str)
        
        print(f"\n✅ REPORTE COMPLETO GENERADO:")
        print(f"   📊 Evaluación detallada: detailed_class_evaluation_report.png")
        print(f"   📈 Métricas por clase: class_metrics.json")
        print(f"   🎯 Umbrales adaptativos: adaptive_thresholds.json")
        print(f"   📋 Reporte completo: complete_class_evaluation_report.json")
        
        return complete_report

def main():
    """Función principal"""
    evaluator = DetailedClassEvaluator()
    results = evaluator.generate_complete_report(samples_per_class=30)
    
    return results

if __name__ == "__main__":
    results = main()