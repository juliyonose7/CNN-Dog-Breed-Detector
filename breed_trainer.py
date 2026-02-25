"""
Dog Breed Classification Trainer
=================================

A specialized training module for multi-class dog breed classification.
Optimized for efficient training with advanced techniques on AMD 7800X3D hardware.

Features:
    - Multiple architecture support (EfficientNet, ResNet, ConvNeXt)
    - Mixed precision training for improved performance
    - Advanced learning rate schedulers (OneCycleLR, CosineAnnealing)
    - Adaptive data augmentation pipeline
    - Detailed metrics including Top-1 and Top-5 accuracy
    - Confusion matrix visualization
    - Early stopping with model checkpointing
    - AMD 7800X3D specific optimizations

Supported Architectures:
    - efficientnet_b3: Balanced accuracy and speed
    - resnet50: Classic robust architecture  
    - convnext_small: Modern Vision Transformer-inspired architecture

Usage:
    Run directly: python breed_trainer.py
    
Author: NotDog YesDog Team
Optimized for: AMD 7800X3D (16 threads)
"""

# === IMPORTS ===

# Operating system and utility imports
import os                    # Operating system operations
import time                  # Time measurement
import json                  # JSON data handling
import random                # Random number generation
from pathlib import Path     # Modern path handling
from typing import Dict, List, Tuple  # Type annotations

# Data analysis and visualization imports
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Plotting and visualizations
import seaborn as sns        # Statistical visualizations
import pandas as pd          # DataFrame manipulation
from tqdm import tqdm        # Progress bars

# PyTorch deep learning imports
import torch                 # Main framework
import torch.nn as nn        # Neural network modules
import torch.optim as optim  # Optimizers
import torch.nn.functional as F  # Activation functions
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR  # LR schedulers
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
from torchvision import models  # Pretrained models

# Metrics and evaluation imports
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

# === MODEL ARCHITECTURE ===


class AdvancedBreedClassifier(nn.Module):
    """Advanced neural network model for multi-class dog breed classification.
    
    Supports multiple backbone architectures with a custom classification head
    featuring dropout regularization and batch normalization.
    
    Supported backbones:
        - efficientnet_b3: Balanced accuracy and speed
        - resnet50: Classic robust architecture
        - convnext_small: Modern Vision Transformer-inspired design
    
    Args:
        num_classes: Number of dog breeds to classify.
        model_name: Backbone architecture name.
        pretrained: Whether to use ImageNet pretrained weights.
    
    Attributes:
        backbone: Feature extraction network.
        classifier: Custom classification head.
    """
    
    def __init__(self, num_classes=50, model_name='efficientnet_b3', pretrained=True):
        """Initialize the classifier with the selected architecture.
        
        Args:
            num_classes: Number of breed classes to classify.
            model_name: Architecture to use ('efficientnet_b3', 'resnet50', 'convnext_small').
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super(AdvancedBreedClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Create backbone based on specified architecture
        # Each architecture has different feature dimensions
        if model_name == 'efficientnet_b3':
            # EfficientNet-B3: balanced between accuracy and speed
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remove original classifier
            
        elif model_name == 'resnet50':
            # ResNet-50: classic robust architecture
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            
        elif model_name == 'convnext_small':
            # ConvNeXt: modern architecture inspired by Vision Transformers
            self.backbone = models.convnext_small(pretrained=pretrained)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()  # Remove original classifier
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classification head with progressive dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize classifier weights using Kaiming initialization.
        
        Uses Kaiming (He) initialization for linear layers and
        constant initialization for batch normalization layers.
        """
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Tensor: Class logits of shape (batch_size, num_classes).
        """
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract feature representations without classification.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Tensor: Feature vectors of shape (batch_size, feature_dim).
        """
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        return features

# === TRAINER CLASS ===


class BreedTrainer:
    """Advanced training manager for dog breed classification.
    
    Provides a complete training pipeline with:
    - Configurable backbone architectures
    - OneCycleLR learning rate scheduling
    - Top-1 and Top-5 accuracy metrics
    - Early stopping and model checkpointing
    - Training visualization and reports
    - AMD 7800X3D optimizations
    
    Args:
        model_name: Backbone architecture name.
        num_classes: Number of breed classes.
        device: Computing device ('cpu' or 'cuda').
    
    Attributes:
        model: The neural network model.
        training_history: Dictionary tracking metrics per epoch.
        breed_names: Mapping from class indices to breed names.
    """
    
    def __init__(self, model_name='efficientnet_b3', num_classes=50, device='cpu'):
        """Initialize the trainer with model and device configuration."""
        self.device = device
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Create model instance
        self.model = AdvancedBreedClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=True
        ).to(device)
        
        # Training state tracking
        self.best_val_acc = 0.0
        self.best_val_top5 = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'train_top5': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5': [],
            'learning_rates': []
        }
        
        # Load breed configuration for class names
        try:
            from breed_processed_data.dataset_config import DATASET_INFO, INDEX_TO_DISPLAY
            self.breed_names = INDEX_TO_DISPLAY
            self.dataset_info = DATASET_INFO
            print(" Breed configuration loaded")
        except ImportError:
            print("  Breed configuration not found")
            self.breed_names = {i: f"Breed_{i}" for i in range(num_classes)}
            self.dataset_info = {}
        
        # Apply AMD 7800X3D optimizations
        self.setup_environment()
    
    def setup_environment(self):
        """Configure the environment for optimal AMD 7800X3D performance.
        
        Sets PyTorch threading and enables MKL-DNN optimizations.
        """
        # Configure PyTorch threading for CPU
        torch.set_num_threads(16)
        torch.set_num_interop_threads(16)
        
        # Enable MKL-DNN optimizations if available
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        print(" Environment optimized for 7800X3D")
    
    def setup_training(self, train_loader, val_loader, learning_rate=1e-3, weight_decay=1e-4):
        """Configure optimizer, scheduler, and loss function.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            learning_rate: Base learning rate.
            weight_decay: L2 regularization coefficient.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # AdamW optimizer with optimized configuration
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # OneCycleLR scheduler for optimal learning rate scheduling
        total_steps = len(train_loader) * 40  # 40 epochs assumed
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 5,  # Peak learning rate
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup phase
            div_factor=25,  # Initial LR = max_lr/25
            final_div_factor=10000  # Final LR = max_lr/10000
        )
        
        # Cross-entropy loss with label smoothing for regularization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Gradient scaler for mixed precision (disabled for CPU)
        self.scaler = GradScaler(enabled=False)  # CPU does not support AMP
        
        print("  Training configuration ready")
        print(f"    Learning rate: {learning_rate}")
        print(f"    Total steps: {total_steps:,}")
        print(f"    Classes: {self.num_classes}")
    
    def calculate_metrics(self, outputs, targets):
        """Calculate Top-1 and Top-5 accuracy metrics.
        
        Args:
            outputs: Model output logits of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).
            
        Returns:
            tuple: (top1_accuracy, top5_accuracy) as floats.
        """
        predictions = torch.argmax(outputs, dim=1)
        
        # Top-1 accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        
        # Top-5 accuracy
        _, top5_pred = torch.topk(outputs, min(5, outputs.size(1)), dim=1)
        top5_correct = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))
        top5_accuracy = top5_correct.any(dim=1).float().mean().item()
        
        return accuracy, top5_accuracy
    
    def train_epoch(self, epoch):
        """Train the model for one epoch.
        
        Args:
            epoch: Current epoch number (0-indexed).
            
        Returns:
            tuple: (avg_loss, avg_accuracy, avg_top5, learning_rate)
        """
        self.model.train()
        
        running_loss = 0.0
        running_acc = 0.0
        running_top5 = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=False):  # CPU does not support autocast
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Calculate batch metrics
            acc, top5_acc = self.calculate_metrics(outputs, targets)
            
            # Accumulate running metrics
            running_loss += loss.item()
            running_acc += acc
            running_top5 += top5_acc
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.3f}',
                'Top5': f'{top5_acc:.3f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calculate epoch averages
        avg_loss = running_loss / len(self.train_loader)
        avg_acc = running_acc / len(self.train_loader)
        avg_top5 = running_top5 / len(self.train_loader)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Save to training history
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['train_acc'].append(avg_acc)
        self.training_history['train_top5'].append(avg_top5)
        self.training_history['learning_rates'].append(current_lr)
        
        return avg_loss, avg_acc, avg_top5, current_lr
    
    def validate_epoch(self):
        """Validate the model on the validation set.
        
        Returns:
            tuple: (avg_loss, avg_accuracy, avg_top5, predictions, targets)
        """
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        running_top5 = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                acc, top5_acc = self.calculate_metrics(outputs, targets)
                
                running_loss += loss.item()
                running_acc += acc
                running_top5 += top5_acc
                
                # Store predictions for confusion matrix
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch averages
        avg_loss = running_loss / len(self.val_loader)
        avg_acc = running_acc / len(self.val_loader)
        avg_top5 = running_top5 / len(self.val_loader)
        
        # Save to training history
        self.training_history['val_loss'].append(avg_loss)
        self.training_history['val_acc'].append(avg_acc)
        self.training_history['val_top5'].append(avg_top5)
        
        return avg_loss, avg_acc, avg_top5, all_predictions, all_targets
    
    def save_checkpoint(self, epoch, save_path, is_best=False):
        """Save a training checkpoint.
        
        Args:
            epoch (int): Current epoch number.
            save_path (str or Path): Directory to save checkpoint.
            is_best (bool): Whether this is the best model so far.
        
        Returns:
            Path: Path to the saved checkpoint file.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_top5': self.best_val_top5,
            'training_history': self.training_history,
            'model_config': {
                'num_classes': self.num_classes,
                'model_name': self.model_name,
                'breed_names': self.breed_names
            }
        }
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = save_path / f'breed_model_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = save_path / 'best_breed_model.pth'
            torch.save(checkpoint, best_path)
            print(f" Best model saved: {best_path}")
        
        return checkpoint_path
    
    def create_confusion_matrix(self, predictions, targets, save_path=None):
        """Generate and save confusion matrix visualization.
        
        Creates a normalized confusion matrix heatmap for the top 20
        performing classes to maintain readability.
        
        Args:
            predictions (list): Model predictions.
            targets (list): Ground truth labels.
            save_path (str or Path, optional): Path to save the figure.
        """
        cm = confusion_matrix(targets, predictions)
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # Normalize confusion matrix by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Select top 20 classes by diagonal accuracy for visualization
        top_20_indices = np.argsort(np.diag(cm))[-20:]
        cm_subset = cm_normalized[np.ix_(top_20_indices, top_20_indices)]
        
        # Plot heatmap
        sns.heatmap(cm_subset, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=[self.breed_names.get(i, f'Class_{i}') for i in top_20_indices],
                   yticklabels=[self.breed_names.get(i, f'Class_{i}') for i in top_20_indices])
        
        plt.title('Confusion Matrix - Top 20 Classes', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def plot_training_history(self, save_path=None):
        """Plot training history metrics.
        
        Creates a 2x2 grid of plots showing:
        - Training/validation loss over epochs
        - Top-1 accuracy over epochs
        - Top-5 accuracy over epochs  
        - Learning rate schedule
        
        Args:
            save_path (str or Path, optional): Path to save the figure.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation')
        ax1.set_title('Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Top-1 Accuracy plot
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training')
        ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation')
        ax2.set_title('Top-1 Accuracy per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Top-5 Accuracy plot
        ax3.plot(epochs, self.training_history['train_top5'], 'b-', label='Training')
        ax3.plot(epochs, self.training_history['val_top5'], 'r-', label='Validation')
        ax3.set_title('Top-5 Accuracy per Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Learning Rate plot
        ax4.plot(epochs, self.training_history['learning_rates'], 'g-')
        ax4.set_title('Learning Rate per Epoch')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Training history saved: {save_path}")
        
        plt.close()
        
        plt.close()
    
    def train_model(self, num_epochs=30, save_path='./breed_models', patience=7):
        """Execute the complete training loop.
        
        Args:
            num_epochs (int): Maximum number of training epochs.
            save_path (str): Directory to save models and visualizations.
            patience (int): Early stopping patience (epochs without improvement).
        
        Returns:
            dict: Training results containing:
                - best_val_acc: Best validation accuracy achieved
                - best_val_top5: Best validation top-5 accuracy
                - training_time: Total training time in seconds
                - total_epochs: Number of epochs completed
                - save_path: Path where models were saved
        """
        start_time = time.time()
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        print(" STARTING BREED TRAINING")
        print("="*60)
        print(f" Epochs: {num_epochs}")
        print(f"  Classes: {self.num_classes}")
        print(f" Model: {self.model_name}")
        print(f" Device: {self.device}")
        
        # Counter for early stopping
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n EPOCH {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Training phase
            train_loss, train_acc, train_top5, current_lr = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_top5, predictions, targets = self.validate_epoch()
            
            # Display results
            print(f" Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, Top5: {train_top5:.3f}")
            print(f" Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Top5: {val_top5:.3f}")
            print(f" Learning Rate: {current_lr:.2e}")
            
            # Check if this is the best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_top5 = val_top5
                patience_counter = 0
                print(f" New best model! Acc: {val_acc:.3f}, Top5: {val_top5:.3f}")
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, save_path, is_best)
            
            # Early stopping check
            if patience_counter >= patience:
                print(f" Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Training completed
        elapsed_time = time.time() - start_time
        
        # Create final visualizations
        self.plot_training_history(save_path / 'training_history.png')
        self.create_confusion_matrix(predictions, targets, save_path / 'confusion_matrix.png')
        
        # Save final configuration
        final_config = {
            'training_completed': True,
            'total_epochs': epoch + 1,
            'best_val_acc': self.best_val_acc,
            'best_val_top5': self.best_val_top5,
            'training_time_hours': elapsed_time / 3600,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'breed_names': self.breed_names
        }
        
        with open(save_path / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n TRAINING COMPLETED")
        print("="*60)
        print(f"  Total time: {elapsed_time/3600:.2f} hours")
        print(f" Best Accuracy: {self.best_val_acc:.3f}")
        print(f" Best Top-5: {self.best_val_top5:.3f}")
        print(f" Models saved in: {save_path}")
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_val_top5': self.best_val_top5,
            'training_time': elapsed_time,
            'total_epochs': epoch + 1,
            'save_path': save_path
        }

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for breed model training.
    
    Initializes the training pipeline by:
    1. Detecting and configuring the compute device
    2. Loading the preprocessed breed dataset
    3. Creating and configuring the BreedTrainer
    4. Executing the training loop
    
    Returns:
        dict: Training results or None if an error occurred.
    """
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Using device: {device}")
    
    # Load preprocessed data
    try:
        from breed_preprocessor import BreedDatasetPreprocessor
        
        yesdog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\YESDOG"
        preprocessor = BreedDatasetPreprocessor(yesdog_path)
        
        print(" Loading preprocessed dataset...")
        # Import dataset configuration
        from breed_processed_data.dataset_config import DATASET_INFO
        
        # Create DataLoaders using the preprocessor
        results = preprocessor.run_complete_preprocessing(target_samples_per_class=200)
        data_loaders = results['data_loaders']
        
        print(f" Dataset loaded:")
        print(f"     Train: {len(data_loaders['train'].dataset)} samples")
        print(f"    Val: {len(data_loaders['val'].dataset)} samples")
        
    except Exception as e:
        print(f" Error loading dataset: {e}")
        print(" Run breed_preprocessor.py first")
        return None
    
    # Create trainer
    trainer = BreedTrainer(
        model_name='efficientnet_b3',
        num_classes=50,
        device=device
    )
    
    # Configure training
    trainer.setup_training(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # Train model
    results = trainer.train_model(
        num_epochs=30,
        save_path='./breed_models',
        patience=7
    )
    
    return results

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    results = main()
    
    if results:
        print(f"\n Training successful!")
        print(f" Best accuracy: {results['best_val_acc']:.3f}")
        print(f"  Total time: {results['training_time']/3600:.2f} hours")
    else:
        print(" Training error")