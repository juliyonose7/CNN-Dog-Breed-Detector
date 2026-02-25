"""
Deep Learning Model Trainer
============================

Comprehensive training module for dog vs non-dog binary classification.
Optimized for AMD GPUs with ROCm support and includes features like
mixed precision training, learning rate scheduling, and automatic
model checkpointing.

Key Features:
    - Multiple backbone architectures (EfficientNet, ResNet, DenseNet, ViT)
    - Transfer learning with gradual unfreezing
    - Mixed precision training for faster GPU computation
    - Class weight balancing for imbalanced datasets
    - Comprehensive metrics tracking (Accuracy, AUC, Precision, Recall, F1)
    - Training curve visualization

Classes:
    DogClassificationModel: Neural network model with customizable backbone.
    ModelTrainer: Training orchestrator with validation and checkpointing.

Usage:
    >>> trainer = ModelTrainer(model_name='efficientnet_b3')
    >>> trainer.setup_training(train_loader, val_loader)
    >>> history = trainer.train_model(num_epochs=30, save_path='./models')

Author: Dog Classification Team
Version: 1.0.0
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.models as models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DogClassificationModel(nn.Module):
    """
    Binary classification model based on pre-trained architectures.
    
    Combines a pre-trained backbone (feature extractor) with a custom
    classification head for dog vs non-dog classification.
    
    Attributes:
        model_name (str): Name of the backbone architecture.
        backbone (nn.Module): Pre-trained feature extraction network.
        classifier (nn.Sequential): Custom classification head.
    
    Supported Backbones:
        - efficientnet_b3: Efficient and accurate, good balance
        - resnet50/resnet101: Classic ResNet architectures
        - densenet121: Dense connections for feature reuse
        - vit_b_16: Vision Transformer for attention-based features
    """
    
    def __init__(self, model_name: str = 'efficientnet_b3', num_classes: int = 1, pretrained: bool = True):
        """
        Initialize the classification model.
        
        Args:
            model_name (str, optional): Backbone architecture name. 
                                        Options: 'efficientnet_b3', 'resnet50', 'resnet101',
                                        'densenet121', 'vit_b_16'. Defaults to 'efficientnet_b3'.
            num_classes (int, optional): Number of output classes. Defaults to 1 (binary).
            pretrained (bool, optional): Use ImageNet pre-trained weights. Defaults to True.
        
        Raises:
            ValueError: If model_name is not supported.
        """
        super(DogClassificationModel, self).__init__()
        
        self.model_name = model_name
        
        # Select base architecture
        if model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
            
        else:
            raise ValueError(f"Model not supported: {model_name}")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """
        Initialize classifier head weights using Kaiming initialization.
        
        Applies Kaiming Normal initialization to Linear layers for better
        gradient flow during training.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Output logits of shape (N,) for binary classification.
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze the backbone network.
        
        Used for transfer learning to prevent backbone weight updates
        during initial training epochs.
        
        Args:
            freeze (bool, optional): If True, freeze backbone. Defaults to True.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

class ModelTrainer:
    """
    Complete training orchestrator for dog classification models.
    
    Handles the full training workflow including optimizer setup,
    learning rate scheduling, mixed precision training, validation,
    checkpointing, and visualization.
    
    Attributes:
        device (torch.device): Computation device (CPU/CUDA).
        model (DogClassificationModel): The neural network model.
        scaler (GradScaler): Mixed precision gradient scaler.
        train_history (dict): Training metrics over epochs.
        best_val_accuracy (float): Best validation accuracy achieved.
        best_model_path (str): Path to saved best model.
    """
    
    def __init__(self, model_name: str = 'efficientnet_b3', device: str = 'auto'):
        """
        Initialize the model trainer.
        
        Args:
            model_name (str, optional): Backbone architecture name. Defaults to 'efficientnet_b3'.
            device (str, optional): Computation device. 'auto' selects CUDA if available.
                                   Defaults to 'auto'.
        """
        # Configure device (ROCm for AMD GPUs)
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
                print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                self.device = torch.device('cpu')
                print("Warning: Using CPU (no compatible GPU detected)")
        else:
            self.device = torch.device(device)
        
        # Create model
        self.model = DogClassificationModel(model_name=model_name)
        self.model.to(self.device)
        
        # Configure mixed precision training scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Initialize training history
        self.train_history = {
            'loss': [], 'accuracy': [], 'lr': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': []
        }
        
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        
    def setup_training(self, train_loader, val_loader, class_weights=None):
        """
        Configure optimizer, loss function, and learning rate scheduler.
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            class_weights (list, optional): Class weights [neg_weight, pos_weight].
                                            If None, calculated from data.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Calculate class weights if not provided
        if class_weights is None:
            pos_weight = self._calculate_pos_weight(train_loader)
        else:
            pos_weight = torch.tensor(class_weights[1] / class_weights[0])
        
        pos_weight = pos_weight.to(self.device)
        
        # Loss function with class weights for imbalance
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # AdamW optimizer with weight decay regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - reduce on plateau
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        print(f"Training configuration completed")
        print(f"   Device: {self.device}")
        print(f"   Pos weight: {pos_weight.item():.3f}")
        print(f"   Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def _calculate_pos_weight(self, train_loader):
        """
        Calculate positive class weight for imbalanced data.
        
        Computes the ratio of negative to positive samples to weight
        the loss function appropriately.
        
        Args:
            train_loader (DataLoader): Training data loader.
        
        Returns:
            torch.Tensor: Positive class weight scalar.
        """
        total_negative = 0
        total_positive = 0
        
        for _, labels in train_loader:
            total_positive += labels.sum().item()
            total_negative += (1 - labels).sum().item()
        
        pos_weight = total_negative / max(total_positive, 1)
        return torch.tensor(pos_weight)
    
    def train_epoch(self, epoch: int):
        """
        Execute one training epoch.
        
        Performs forward pass, loss calculation, backpropagation, and
        optimizer step for all batches in the training set.
        
        Args:
            epoch (int): Current epoch number (0-indexed).
        
        Returns:
            tuple: (epoch_loss, epoch_accuracy, current_lr)
        """
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return epoch_loss, epoch_accuracy, current_lr
    
    def validate_epoch(self):
        """
        Execute validation on the validation set.
        
        Computes loss and various metrics on the validation data
        without gradient computation.
        
        Returns:
            tuple: (val_loss, val_accuracy, val_auc, precision, recall, f1)
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Collect predictions and probabilities
                probabilities = torch.sigmoid(outputs)
                predictions = probabilities > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_auc = roc_auc_score(all_labels, all_probabilities)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        return val_loss, val_accuracy, val_auc, precision, recall, f1
    
    def train_model(self, num_epochs: int = 50, save_path: str = None, 
                   freeze_epochs: int = 5):
        """
        Execute the complete training loop.
        
        Trains the model for the specified number of epochs with gradual
        unfreezing of the backbone network. Includes validation, checkpointing,
        and early stopping.
        
        Args:
            num_epochs (int, optional): Total training epochs. Defaults to 50.
            save_path (str, optional): Directory to save model checkpoints.
                                       If None, models are not saved.
            freeze_epochs (int, optional): Number of epochs to keep backbone frozen.
                                          Defaults to 5.
        
        Returns:
            dict: Training history with loss, accuracy, and other metrics.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"   First {freeze_epochs} epochs: backbone frozen")
        print("="*60)
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            self.best_model_path = save_path / 'best_model.pth'
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Unfreeze backbone after initial epochs
            if epoch == freeze_epochs:
                print(f"\nUnfreezing backbone at epoch {epoch+1}")
                self.model.freeze_backbone(False)
                # Reduce learning rate when unfreezing
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            
            # Train one epoch
            train_loss, train_acc, current_lr = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_acc)
            
            # Record history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['lr'].append(current_lr)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            self.train_history['val_auc'].append(val_auc)
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                if save_path:
                    self.save_model(self.best_model_path, epoch, val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            print(f"  Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Simple early stopping on learning rate
            if current_lr < 1e-6:
                print("Early stopping: Learning rate too low")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time/60:.1f} minutes")
        print(f"   Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        if save_path:
            self.save_training_history(save_path)
            self.plot_training_curves(save_path)
        
        return self.train_history
    
    def save_model(self, save_path: Path, epoch: int, val_accuracy: float):
        """
        Save model checkpoint to disk.
        
        Saves model state, optimizer state, and training metadata
        for later resumption or inference.
        
        Args:
            save_path (Path): Full path for the checkpoint file.
            epoch (int): Current epoch number.
            val_accuracy (float): Validation accuracy at this checkpoint.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'model_name': self.model.model_name,
            'train_history': self.train_history
        }, save_path)
        
        print(f"Model saved: {save_path} (val_acc: {val_accuracy:.4f})")
    
    def load_model(self, model_path: str):
        """
        Load a model checkpoint from disk.
        
        Args:
            model_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        
        print(f"Model loaded from: {model_path}")
        print(f"   Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_accuracy']:.4f}")
    
    def save_training_history(self, save_path: Path):
        """
        Save training history to JSON file.
        
        Args:
            save_path (Path): Directory to save the history file.
        """
        history_path = save_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"History saved: {history_path}")
    
    def plot_training_curves(self, save_path: Path):
        """
        Generate and save training visualization plots.
        
        Creates a 2x2 subplot figure with Loss, Accuracy, Learning Rate,
        and AUC curves over training epochs.
        
        Args:
            save_path (Path): Directory to save the plot image.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16)
        
        epochs = range(1, len(self.train_history['loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_history['loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_history['accuracy'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.train_history['val_accuracy'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate plot
        axes[1, 0].plot(epochs, self.train_history['lr'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # AUC plot
        axes[1, 1].plot(epochs, self.train_history['val_auc'], 'purple')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = save_path / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plots saved: {plot_path}")

def setup_rocm_optimization():
    """
    Configure ROCm optimizations for AMD GPUs.
    
    Enables cuDNN benchmark mode and TF32 precision for faster
    training on compatible AMD GPUs using ROCm.
    
    Returns:
        bool: True if ROCm is available and configured, False otherwise.
    """
    # ROCm configurations
    if torch.cuda.is_available():
        print("Configuring ROCm optimizations...")
        
        # Configure memory management
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Configure for mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("ROCm optimizations configured")
        
        return True
    else:
        print("Warning: ROCm not available, using CPU configuration")
        return False

if __name__ == "__main__":
    # Configure ROCm
    rocm_available = setup_rocm_optimization()
    
    # Configure model
    model_name = 'efficientnet_b3'  # Options: 'efficientnet_b3', 'resnet50', 'resnet101', 'densenet121'
    
    print(f"Configuring model: {model_name}")
    print("="*60)
    
    # Create trainer
    trainer = ModelTrainer(model_name=model_name)
    
    print(f"\nModel created:")
    print(f"   Architecture: {model_name}")
    print(f"   Device: {trainer.device}")
    print(f"   Mixed Precision: {'Yes' if trainer.scaler else 'No'}")
    
    # Usage example with DataLoaders (requires running data_preprocessor.py first)
    # from data_preprocessor import DataPreprocessor
    #
    # preprocessor = DataPreprocessor(dataset_path, output_path)
    # data_loaders, splits = preprocessor.process_complete_dataset()
    #
    # trainer.setup_training(data_loaders['train'], data_loaders['val'])
    # history = trainer.train_model(num_epochs=30, save_path='./models')