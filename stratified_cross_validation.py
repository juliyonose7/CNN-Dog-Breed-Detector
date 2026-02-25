# !/usr/bin/env python3
"""
Technical documentation in English.
============================================================

Technical documentation in English.
Technical documentation in English.

Autor: System IA
Fecha: 2024
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm

class BalancedDogDataset(Dataset):
    """Dataset personalizado for el dataset balanced"""
    
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load el dataset balanced"""
        print(f"📁 Cargando dataset desde: {self.dataset_path}")
        
        # Obtener all las classes (directorios)
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        print(f"📋 Clases encontradas: {len(self.classes)}")
        
        # Load all las images
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Buscar images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(class_dir.glob(ext)))
            
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
        
        print(f"📊 Total de imágenes: {len(self.samples):,}")
        
        # Implementation note.
        class_counts = Counter([sample[1] for sample in self.samples])
        print(f"📈 Distribución por clase:")
        for class_name, class_idx in self.class_to_idx.items():
            count = class_counts[class_idx]
            print(f"   {class_name:25} | {count:4d} imágenes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"❌ Error cargando imagen {img_path}: {e}")
            # Implementation note.
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            return image, label

class StratifiedCrossValidator:
    def __init__(self, dataset_path: str, workspace_path: str, n_folds: int = 5):
        self.dataset_path = Path(dataset_path)
        self.workspace_path = Path(workspace_path)
        self.n_folds = n_folds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️ Dispositivo: {self.device}")
        
        # Configurar transformaciones
        self.setup_transforms()
        
        # Load dataset
        self.load_dataset()
        
        # Resultados por fold
        self.fold_results = []
        
    def setup_transforms(self):
        """Configura las transformaciones for training y validation"""
        
        # Transformaciones for training (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transformaciones for validation (without augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_dataset(self):
        """Load el dataset balanced"""
        self.dataset = BalancedDogDataset(self.dataset_path, transform=self.val_transform)
        
        if len(self.dataset) == 0:
            raise ValueError("❌ Dataset vacío")
        
        # Extraer labels for stratified split
        self.labels = np.array([sample[1] for sample in self.dataset.samples])
        self.n_classes = len(self.dataset.classes)
        
        print(f"📊 Dataset cargado:")
        print(f"   Clases: {self.n_classes}")
        print(f"   Muestras: {len(self.dataset):,}")
        print(f"   Folds: {self.n_folds}")
    
    def create_model(self):
        """Crea un model ResNet50 for classification"""
        import torchvision.models as models
        
        model = models.resnet50(pretrained=True)
        
        # Congelar capas base (feature extraction)
        for param in model.parameters():
            param.requires_grad = False
        
        # Reemplazar clasificador
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.n_classes)
        )
        
        # Only entrenar el clasificador
        for param in model.fc.parameters():
            param.requires_grad = True
        
        return model.to(self.device)
    
    def train_fold(self, model, train_loader, val_loader, fold_num, epochs=10):
        """Technical documentation in English."""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        print(f"\n🚀 Entrenando Fold {fold_num + 1}/{self.n_folds}")
        print("-" * 40)
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/10 - Train', 
                             leave=False, disable=True)  # Silencioso for no saturar output
            
            for batch_idx, (inputs, targets) in enumerate(train_pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += targets.size(0)
                correct_train += predicted.eq(targets).sum().item()
                
                if batch_idx % 20 == 0:  # Mostrar progreso cada 20 batches
                    print(f"      Batch {batch_idx:3d}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Acc: {100.*correct_train/total_train:.1f}%")
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct_train / total_train
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += targets.size(0)
                    correct_val += predicted.eq(targets).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100. * correct_val / total_val
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            print(f"   Época {epoch+1:2d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold_{fold_num}.pth')
            
            scheduler.step(val_loss)
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def evaluate_fold(self, model, val_loader, fold_num):
        """Technical documentation in English."""
        
        # Load best model of the fold
        model.load_state_dict(torch.load(f'best_model_fold_{fold_num}.pth', map_location=self.device))
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Evaluando Fold {fold_num+1}'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                _, predictions = outputs.max(1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Implementation note.
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        # Implementation note.
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=self.dataset.classes,
            output_dict=True, zero_division=0
        )
        
        # Implementation note.
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        print(f"✅ Fold {fold_num+1} - Accuracy: {accuracy:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        return {
            'fold': fold_num + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_report': class_report,
            'conf_matrix': conf_matrix.tolist(),
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def run_stratified_kfold_validation(self, epochs_per_fold=10):
        """Ejecuta validation cruzada estratificada completa"""
        
        print(f"\n🔍 INICIANDO VALIDACIÓN CRUZADA ESTRATIFICADA")
        print("=" * 70)
        
        # Crear StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold_num, (train_indices, val_indices) in enumerate(skf.split(self.labels, self.labels)):
            
            print(f"\n📋 FOLD {fold_num + 1}/{self.n_folds}")
            print("-" * 50)
            print(f"   Train samples: {len(train_indices):,}")
            print(f"   Val samples: {len(val_indices):,}")
            
            # Implementation note.
            train_labels = self.labels[train_indices]
            val_labels = self.labels[val_indices]
            
            train_dist = Counter(train_labels)
            val_dist = Counter(val_labels)
            
            print(f"   Distribución estratificada verificada:")
            for class_idx in range(min(5, self.n_classes)):  # Mostrar only 5 primeras classes
                class_name = self.dataset.classes[class_idx]
                train_pct = (train_dist[class_idx] / len(train_indices)) * 100
                val_pct = (val_dist[class_idx] / len(val_indices)) * 100
                print(f"      {class_name:20} | Train: {train_pct:.1f}% | Val: {val_pct:.1f}%")
            
            # Implementation note.
            train_dataset = BalancedDogDataset(self.dataset_path, transform=self.train_transform)
            val_dataset = BalancedDogDataset(self.dataset_path, transform=self.val_transform)
            
            # Crear samplers
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            
            # Crear data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=32, 
                sampler=train_sampler,
                num_workers=2,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=64, 
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True
            )
            
            # Crear y entrenar model
            model = self.create_model()
            
            # Entrenar fold
            training_history = self.train_fold(
                model, train_loader, val_loader, fold_num, epochs_per_fold
            )
            
            # Evaluar fold
            evaluation_results = self.evaluate_fold(model, val_loader, fold_num)
            
            # Combinar resultados
            fold_result = {
                **evaluation_results,
                'training_history': training_history
            }
            
            fold_results.append(fold_result)
            
            # Limpiar memoria
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.fold_results = fold_results
        return self.analyze_kfold_results()
    
    def analyze_kfold_results(self):
        """Analiza los resultados de all los folds"""
        
        print(f"\n📊 ANÁLISIS DE RESULTADOS K-FOLD")
        print("=" * 70)
        
        # Implementation note.
        accuracies = [result['accuracy'] for result in self.fold_results]
        precisions = [result['precision'] for result in self.fold_results]
        recalls = [result['recall'] for result in self.fold_results]
        f1_scores = [result['f1'] for result in self.fold_results]
        
        # Implementation note.
        stats = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'min': np.min(precisions),
                'max': np.max(precisions)
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'min': np.min(recalls),
                'max': np.max(recalls)
            },
            'f1': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores)
            }
        }
        
        print(f"📈 ESTADÍSTICAS GLOBALES ({self.n_folds}-FOLD):")
        for metric, values in stats.items():
            print(f"   {metric.upper():10} | "
                  f"Media: {values['mean']:.4f} ± {values['std']:.4f} | "
                  f"Rango: [{values['min']:.4f}, {values['max']:.4f}]")
        
        # Implementation note.
        class_metrics = self.analyze_per_class_performance()
        
        # Crear visualizaciones
        self.create_kfold_visualizations(stats, class_metrics)
        
        # Save reporte completo
        final_report = {
            'timestamp': str(np.datetime64('now')),
            'n_folds': self.n_folds,
            'n_classes': self.n_classes,
            'n_samples': len(self.dataset),
            'global_stats': stats,
            'class_metrics': class_metrics,
            'fold_results': self.fold_results
        }
        
        with open('stratified_kfold_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ VALIDACIÓN CRUZADA COMPLETADA")
        print(f"   📊 Accuracy promedio: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
        print(f"   📁 Reporte guardado: stratified_kfold_validation_report.json")
        
        return final_report
    
    def analyze_per_class_performance(self):
        """Analiza el rendimiento promedio por class across folds"""
        
        class_performance = defaultdict(list)
        
        for fold_result in self.fold_results:
            class_report = fold_result['class_report']
            
            for class_name in self.dataset.classes:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    class_performance[class_name].append({
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1-score': metrics['f1-score']
                    })
        
        # Calcular promedios y desviaciones
        class_avg_metrics = {}
        
        for class_name, fold_metrics in class_performance.items():
            precisions = [m['precision'] for m in fold_metrics]
            recalls = [m['recall'] for m in fold_metrics]
            f1_scores = [m['f1-score'] for m in fold_metrics]
            
            class_avg_metrics[class_name] = {
                'precision': {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions)
                },
                'recall': {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls)
                },
                'f1': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores)
                }
            }
        
        # Implementation note.
        f1_means = [(name, metrics['f1']['mean']) for name, metrics in class_avg_metrics.items()]
        f1_means.sort(key=lambda x: x[1])
        
        print(f"\n🎯 RENDIMIENTO POR CLASE (Promedio {self.n_folds}-fold):")
        print(f"   🚨 5 CLASES MÁS PROBLEMÁTICAS:")
        for i, (class_name, f1_mean) in enumerate(f1_means[:5], 1):
            metrics = class_avg_metrics[class_name]
            print(f"      {i}. {class_name:25} | F1: {f1_mean:.3f} ± {metrics['f1']['std']:.3f}")
        
        print(f"   ✅ 5 CLASES MEJOR RENDIMIENTO:")
        for i, (class_name, f1_mean) in enumerate(f1_means[-5:], 1):
            metrics = class_avg_metrics[class_name]
            print(f"      {i}. {class_name:25} | F1: {f1_mean:.3f} ± {metrics['f1']['std']:.3f}")
        
        return class_avg_metrics
    
    def create_kfold_visualizations(self, stats, class_metrics):
        """Crea visualizaciones de los resultados K-fold"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'📊 VALIDACIÓN CRUZADA ESTRATIFICADA ({self.n_folds}-FOLD)', fontsize=16, fontweight='bold')
        
        # Implementation note.
        fold_numbers = range(1, self.n_folds + 1)
        accuracies = [result['accuracy'] for result in self.fold_results]
        
        bars1 = ax1.bar(fold_numbers, accuracies, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.axhline(stats['accuracy']['mean'], color='red', linestyle='--', 
                   label=f"Media: {stats['accuracy']['mean']:.3f}")
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('📊 Accuracy por Fold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Implementation note.
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Implementation note.
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        means = [stats[metric.lower()]['mean'] for metric in metrics]
        stds = [stats[metric.lower()]['std'] for metric in metrics]
        
        bars2 = ax2.bar(metrics, means, yerr=stds, capsize=5, 
                       color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'], 
                       alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Score')
        ax2.set_title('📈 Comparación de Métricas (Media ± Std)')
        ax2.grid(True, alpha=0.3)
        
        # Valores en barras
        for bar, mean, std in zip(bars2, means, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Implementation note.
        f1_means = [(name, metrics['f1']['mean']) for name, metrics in class_metrics.items()]
        f1_means.sort(key=lambda x: x[1])
        
        worst_classes = f1_means[:8]  # 8 peores
        worst_names = [name.replace('_', ' ')[:15] for name, _ in worst_classes]
        worst_f1s = [f1 for _, f1 in worst_classes]
        
        bars3 = ax3.barh(range(len(worst_names)), worst_f1s, 
                        color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax3.set_yticks(range(len(worst_names)))
        ax3.set_yticklabels(worst_names, fontsize=9)
        ax3.set_xlabel('F1 Score Promedio')
        ax3.set_title('🚨 Clases Más Problemáticas')
        ax3.grid(True, alpha=0.3)
        
        # Implementation note.
        all_f1s = [metrics['f1']['mean'] for metrics in class_metrics.values()]
        
        ax4.hist(all_f1s, bins=15, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax4.axvline(np.mean(all_f1s), color='red', linestyle='--', 
                   label=f'Media: {np.mean(all_f1s):.3f}')
        ax4.set_xlabel('F1 Score Promedio')
        ax4.set_ylabel('Número de Clases')
        ax4.set_title('📈 Distribución de F1 Scores por Clase')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stratified_kfold_validation_report.png', dpi=300, bbox_inches='tight')
        print("✅ Visualización guardada: stratified_kfold_validation_report.png")

def main():
    """Function principal"""
    workspace_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
    balanced_dataset_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\BALANCED_AUGMENTED_DATASET"
    
    # Verificar that existe el dataset balanced
    if not Path(balanced_dataset_path).exists():
        print(f"❌ Dataset balanceado no encontrado en: {balanced_dataset_path}")
        print(f"   🔧 Ejecuta primero targeted_data_augmentation.py")
        return None
    
    # Crear validador
    validator = StratifiedCrossValidator(
        dataset_path=balanced_dataset_path,
        workspace_path=workspace_path,
        n_folds=5
    )
    
    # Ejecutar validation cruzada
    results = validator.run_stratified_kfold_validation(epochs_per_fold=8)
    
    return results

if __name__ == "__main__":
    results = main()