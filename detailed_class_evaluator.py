# !/usr/bin/env python3
"""
Detailed Per-Class Evaluation Module for Breed Classification.

Provides comprehensive evaluation metrics for multi-class breed classification
models with focus on per-class performance analysis and threshold optimization.

Features:
    - Per-class precision, recall, F1-Score calculation
    - Confusion matrix generation and visualization
    - Optimal threshold computation for each class
    - Problematic class identification
    - Comprehensive visualization reports

Output Files Generated:
    - class_metrics.json: Per-class detailed metrics
    - adaptive_thresholds.json: Optimized prediction thresholds
    - detailed_class_evaluation_report.png: Visual analysis
    - complete_class_evaluation_report.json: Full evaluation report

Author: AI System
Date: 2024
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
    """
    Comprehensive evaluator for breed classification model performance.
    
    Provides detailed per-class metrics, threshold optimization, and
    visualization capabilities for model analysis.
    
    Attributes:
        model_path (str): Path to the trained model checkpoint.
        device (torch.device): Computation device (CPU/CUDA).
        model: Loaded PyTorch model in evaluation mode.
        breed_classes (list): Sorted list of breed class names.
        transform: Image preprocessing transformation pipeline.
    """
    
    def __init__(self, model_path="balanced_models/best_balanced_breed_model_epoch_20_acc_88.1366.pth"):
        """
        Initialize the evaluator with model path.
        
        Args:
            model_path (str): Path to the model checkpoint file.
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and classes will be loaded
        self.model = None
        self.breed_classes = []
        self._load_model_and_classes()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model_and_classes(self):
        """
        Load the trained model and class definitions.
        
        Initializes the BalancedBreedClassifier architecture and loads
        weights from checkpoint. Also loads class names from data directory.
        
        Returns:
            None: Sets self.model and self.breed_classes attributes.
        """
        print(" Loading model for detailed evaluation...")
        
        # Define balanced model architecture
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
            print(f" Model loaded: {checkpoint.get('val_accuracy', 0):.2f}% accuracy")
        else:
            print(f" Model not found: {self.model_path}")
            return
        
        # Load class names from training directory structure
        breed_data_path = "breed_processed_data/train"
        if os.path.exists(breed_data_path):
            self.breed_classes = sorted([d for d in os.listdir(breed_data_path) 
                                       if os.path.isdir(os.path.join(breed_data_path, d))])
            print(f" Loaded {len(self.breed_classes)} classes")
        else:
            print(" Classes directory not found")
    
    def evaluate_all_classes(self, test_data_path="breed_processed_data/val", samples_per_class=100):
        """
        Evaluate model performance across all breed classes.
        
        Performs comprehensive evaluation by running inference on samples
        from each class and computing detailed metrics.
        
        Args:
            test_data_path (str): Path to validation/test data directory.
            samples_per_class (int): Maximum samples to evaluate per class.
            
        Returns:
            dict: Complete evaluation results including:
                - overall_accuracy: Global accuracy
                - class_details: Per-class statistics
                - classification_report: sklearn classification report
                - confusion_matrix: Full confusion matrix
                - problematic_classes: Classes with accuracy < 0.70
                - excellent_classes: Classes with accuracy > 0.95
        """
        print(f" EVALUATING {len(self.breed_classes)} CLASSES...")
        print("="*60)
        
        if not os.path.exists(test_data_path):
            print(f" Test directory not found: {test_data_path}")
            return None
        
        # Collect predictions and true labels
        all_true_labels = []
        all_predicted_labels = []
        all_probabilities = []
        class_details = {}
        
        for class_idx, breed_name in enumerate(self.breed_classes):
            print(f" Evaluating {breed_name} ({class_idx+1}/{len(self.breed_classes)})...")
            
            breed_path = os.path.join(test_data_path, breed_name)
            if not os.path.exists(breed_path):
                print(f"    Directory not found: {breed_path}")
                continue
            
            # Get images from class directory
            image_files = [f for f in os.listdir(breed_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(image_files) == 0:
                print(f"    No images found for {breed_name}")
                continue
            
            # Limit samples for efficiency
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
                    print(f"    Error with {image_file}: {e}")
                    continue
            
            # Calculate class-level statistics
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
                
                print(f"    Accuracy: {breed_accuracy:.3f} | "
                      f"Confidence: {avg_confidence:.3f}±{std_confidence:.3f} | "
                      f"Samples: {len(breed_true_labels)}")
                
                # Add to global lists
                all_true_labels.extend(breed_true_labels)
                all_predicted_labels.extend(breed_predicted_labels)
                all_probabilities.extend(breed_probabilities)
        
        # Calculate global metrics
        print(f"\n CALCULATING GLOBAL METRICS...")
        
        # Generate detailed classification report
        class_names = [self.breed_classes[i] for i in range(len(self.breed_classes))]
        report = classification_report(all_true_labels, all_predicted_labels, 
                                     target_names=class_names, output_dict=True)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        
        # Compile final results
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
        
        # Identify problematic and excellent classes
        problematic_classes = []
        excellent_classes = []
        
        for breed_name, details in class_details.items():
            if details['accuracy'] < 0.7:
                problematic_classes.append((breed_name, details['accuracy']))
            elif details['accuracy'] > 0.95:
                excellent_classes.append((breed_name, details['accuracy']))
        
        problematic_classes.sort(key=lambda x: x[1])  # Sort by accuracy ascending
        excellent_classes.sort(key=lambda x: x[1], reverse=True)  # Descending
        
        evaluation_results['problematic_classes'] = problematic_classes
        evaluation_results['excellent_classes'] = excellent_classes
        
        # Display summary
        print(f"\n EVALUATION SUMMARY:")
        print(f"   Overall Accuracy: {report['accuracy']:.4f}")
        print(f"   Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"   Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
        print(f"   Total samples: {len(all_true_labels):,}")
        
        print(f"\n PROBLEMATIC CLASSES (accuracy < 0.70):")
        for breed, acc in problematic_classes[:10]:
            print(f"   {breed}: {acc:.3f}")
        
        print(f"\n EXCELLENT CLASSES (accuracy > 0.95):")
        for breed, acc in excellent_classes[:10]:
            print(f"   {breed}: {acc:.3f}")
        
        return evaluation_results
    
    def calculate_per_class_metrics(self, evaluation_results):
        """
        Calculate detailed per-class metrics from evaluation results.
        
        Extracts and formats precision, recall, F1-score and other metrics
        for each individual class.
        
        Args:
            evaluation_results (dict): Results from evaluate_all_classes.
            
        Returns:
            dict: Per-class metrics dictionary, also saved to class_metrics.json.
        """
        print(f"\n CALCULATING PRECISE PER-CLASS METRICS...")
        
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
        
        # Save metrics to JSON file
        with open('class_metrics.json', 'w') as f:
            json.dump(per_class_metrics, f, indent=2, default=str)
        
        print(f" Per-class metrics saved: class_metrics.json")
        return per_class_metrics
    
    def compute_optimal_thresholds(self, evaluation_results):
        """
        Compute optimal confidence thresholds for each class.
        
        Analyzes per-class accuracy and confidence statistics to determine
        adaptive thresholds that balance precision and recall.
        
        Strategy:
            - High accuracy (>0.9): Lower threshold for more permissive predictions
            - Medium accuracy (>0.7): Moderate threshold
            - Low accuracy (<0.7): Higher threshold for stricter filtering
        
        Args:
            evaluation_results (dict): Results from evaluate_all_classes.
            
        Returns:
            dict: Optimal threshold for each class, saved to adaptive_thresholds.json.
        """
        print(f"\n CALCULATING OPTIMAL THRESHOLDS PER CLASS...")
        
        # Threshold calculation based on accuracy and confidence statistics
        # Uses dynamic adjustment considering class-specific performance
        # Balances between permissive and restrictive predictions
        
        optimal_thresholds = {}
        
        if evaluation_results and 'class_details' in evaluation_results:
            for breed_name, details in evaluation_results['class_details'].items():
                accuracy = details['accuracy']
                avg_confidence = details['avg_confidence']
                std_confidence = details['std_confidence']
                
                # Dynamic threshold calculation based on performance
                if accuracy > 0.9:
                    # High accuracy class: allow lower threshold
                    threshold = max(0.2, avg_confidence - std_confidence)
                elif accuracy > 0.7:
                    # Medium accuracy: moderate threshold
                    threshold = max(0.3, avg_confidence - 0.5 * std_confidence)
                else:
                    # Problematic class: require higher confidence
                    threshold = max(0.4, avg_confidence)
                
                optimal_thresholds[breed_name] = min(0.8, threshold)
        
        # Save thresholds to file
        with open('adaptive_thresholds.json', 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        
        print(f" Adaptive thresholds calculated for {len(optimal_thresholds)} classes")
        print(f"   Range: {min(optimal_thresholds.values()):.3f} - {max(optimal_thresholds.values()):.3f}")
        print(f" Saved: adaptive_thresholds.json")
        
        return optimal_thresholds
    
    def create_detailed_visualizations(self, evaluation_results):
        """
        Generate comprehensive visualization report for evaluation results.
        
        Creates a multi-panel figure with:
            1. Per-class accuracy bar chart
            2. Accuracy distribution histogram
            3. Confidence vs Accuracy scatter plot
            4. Top 10 problematic classes
            5. Confusion matrix subset
            6. Summary statistics panel
        
        Args:
            evaluation_results (dict): Results from evaluate_all_classes.
            
        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        print(f"\n CREATING DETAILED VISUALIZATIONS...")
        
        if not evaluation_results:
            return None
        
        # Configure matplotlib style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # 1. Accuracy per class bar chart
        class_details = evaluation_results['class_details']
        breeds = list(class_details.keys())
        accuracies = [class_details[breed]['accuracy'] for breed in breeds]
        
        ax = axes[0]
        bars = ax.bar(range(len(breeds)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_title('Accuracy per Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Breeds')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(0, len(breeds), max(1, len(breeds)//10)))
        ax.set_xticklabels([breeds[i] for i in range(0, len(breeds), max(1, len(breeds)//10))], rotation=45, ha='right')
        ax.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
        ax.legend()
        
        # 2. Accuracy distribution histogram
        ax = axes[1]
        ax.hist(accuracies, bins=15, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax.set_title('Accuracy Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Number of Classes')
        ax.axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
        ax.legend()
        
        # 3. Confidence vs Accuracy scatter plot
        confidences = [class_details[breed]['avg_confidence'] for breed in breeds]
        
        ax = axes[2]
        scatter = ax.scatter(confidences, accuracies, c=accuracies, cmap='RdYlGn', s=50, alpha=0.7)
        ax.set_title('Confidence vs Accuracy per Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Average Confidence')
        ax.set_ylabel('Accuracy')
        plt.colorbar(scatter, ax=ax, label='Accuracy')
        
        # Add trend line
        z = np.polyfit(confidences, accuracies, 1)
        p = np.poly1d(z)
        ax.plot(confidences, p(confidences), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()
        
        # 4. Top 10 problematic classes horizontal bar
        sorted_by_accuracy = sorted(class_details.items(), key=lambda x: x[1]['accuracy'])
        worst_10 = sorted_by_accuracy[:10]
        best_10 = sorted_by_accuracy[-10:]
        
        ax = axes[3]
        worst_names = [item[0][:15] for item in worst_10]  # Truncate names
        worst_accs = [item[1]['accuracy'] for item in worst_10]
        bars = ax.barh(range(len(worst_names)), worst_accs, color='lightcoral', edgecolor='darkred')
        ax.set_title('Top 10 Most Problematic Classes', fontweight='bold', fontsize=14)
        ax.set_xlabel('Accuracy')
        ax.set_yticks(range(len(worst_names)))
        ax.set_yticklabels(worst_names, fontsize=10)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, worst_accs)):
            ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=9)
        
        # 5. Confusion matrix subset visualization
        conf_matrix = np.array(evaluation_results['confusion_matrix'])
        
        # Show only first 20x20 for readability
        subset_size = min(20, len(self.breed_classes))
        conf_subset = conf_matrix[:subset_size, :subset_size]
        
        ax = axes[4]
        im = ax.imshow(conf_subset, cmap='Blues', interpolation='nearest')
        ax.set_title(f'Confusion Matrix (First {subset_size} classes)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # 6. Summary statistics text panel
        ax = axes[5]
        ax.axis('off')
        
        # Compile summary statistics
        overall_acc = evaluation_results['overall_accuracy']
        macro_f1 = evaluation_results['macro_avg']['f1-score']
        weighted_f1 = evaluation_results['weighted_avg']['f1-score']
        
        problematic_count = len([acc for acc in accuracies if acc < 0.7])
        excellent_count = len([acc for acc in accuracies if acc > 0.9])
        
        summary_text = f"""
 DETAILED METRICS SUMMARY

 Overall Accuracy: {overall_acc:.4f}
 Macro Avg F1: {macro_f1:.4f}
 Weighted Avg F1: {weighted_f1:.4f}

 Total classes: {len(breeds)}
 Problematic classes (<0.70): {problematic_count}
 Excellent classes (>0.90): {excellent_count}
 Intermediate classes: {len(breeds) - problematic_count - excellent_count}

 Average accuracy: {np.mean(accuracies):.3f}
 Standard deviation: {np.std(accuracies):.3f}
 Minimum accuracy: {min(accuracies):.3f}
 Maximum accuracy: {max(accuracies):.3f}

 Model: Balanced ResNet50
 Unified architecture (no bias)
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('detailed_class_evaluation_report.png', dpi=300, bbox_inches='tight')
        print(f" Visualization saved: detailed_class_evaluation_report.png")
        
        return fig
    
    def generate_complete_report(self, samples_per_class=50):
        """
        Generate a complete per-class evaluation report.
        
        Orchestrates all evaluation methods to produce comprehensive
        analysis including metrics, thresholds, and visualizations.
        
        Args:
            samples_per_class (int): Maximum samples to evaluate per class.
            
        Returns:
            dict: Complete evaluation report with all metrics and analysis.
        """
        print("" * 60)
        print(" GENERATING COMPLETE PER-CLASS METRICS REPORT")
        print("" * 60)
        
        if self.model is None:
            print(" Model not loaded correctly")
            return None
        
        # 1. Evaluate all classes
        evaluation_results = self.evaluate_all_classes(samples_per_class=samples_per_class)
        
        if not evaluation_results:
            print(" Error during evaluation")
            return None
        
        # 2. Calculate detailed per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(evaluation_results)
        
        # 3. Compute optimal thresholds
        optimal_thresholds = self.compute_optimal_thresholds(evaluation_results)
        
        # 4. Create visualizations
        fig = self.create_detailed_visualizations(evaluation_results)
        
        # 5. Save complete report
        complete_report = {
            **evaluation_results,
            'per_class_metrics': per_class_metrics,
            'optimal_thresholds': optimal_thresholds
        }
        
        with open('complete_class_evaluation_report.json', 'w') as f:
            json.dump(complete_report, f, indent=2, default=str)
        
        print(f"\n COMPLETE REPORT GENERATED:")
        print(f"    Detailed evaluation: detailed_class_evaluation_report.png")
        print(f"    Per-class metrics: class_metrics.json")
        print(f"    Adaptive thresholds: adaptive_thresholds.json")
        print(f"    Complete report: complete_class_evaluation_report.json")
        
        return complete_report


def main():
    """
    Main entry point for running the detailed class evaluation.
    
    Creates an evaluator instance and generates the complete report.
    
    Returns:
        dict: Complete evaluation results.
    """
    evaluator = DetailedClassEvaluator()
    results = evaluator.generate_complete_report(samples_per_class=30)
    
    return results


if __name__ == "__main__":
    results = main()