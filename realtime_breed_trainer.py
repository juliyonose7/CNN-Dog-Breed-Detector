# !/usr/bin/env python3
"""
Real-Time Dog Breed Classifier Training with Live Metrics.

This module provides an interactive training environment for multi-class
dog breed classification (50 breeds) with real-time metric visualization
and user-controlled training flow.

Features:
    - Live Train Accuracy, Validation Accuracy, Top-3 Accuracy display
    - Interactive epoch-by-epoch training control
    - Balanced and optimized dataset support
    - ResNet34 backbone for multi-class classification

Controls:
    - Press ENTER after epoch to continue training
    - Press 'q' + ENTER to stop training early

Usage:
    python realtime_breed_trainer.py

Requirements:
    - breed_processed_data/ directory with train/val splits
    - Per-breed subdirectories in each split

Author: AI System
Date: 2024
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm


class RealTimeController:
    """
    Interactive training flow controller with user input handling.
    
    Allows users to interrupt training between epochs by monitoring
    keyboard input in a separate thread.
    
    Attributes:
        should_stop: Flag indicating if training should terminate.
        epoch_complete: Flag indicating current epoch has finished.
    """
    
    def __init__(self):
        """Initialize the controller with default running state."""
        self.should_stop = False
        self.input_thread = None
        self.epoch_complete = False
    
    def start_monitoring(self):
        """Start the input monitoring thread for user control."""
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
    
    def _monitor_input(self):
        """Monitor user input in a separate thread for training control."""
        while not self.should_stop:
            try:
                if self.epoch_complete:
                    print("\n" + "="*70)
                    print("🛑 EPOCH COMPLETED - Continue?")
                    print("   ✅ ENTER = Continue  |  ❌ 'q' + ENTER = Stop")
                    print("="*70)
                    
                    user_input = input(">>> ").strip().lower()
                    if user_input == 'q':
                        print("🛑 Stopping training...")
                        self.should_stop = True
                    else:
                        print("▶️ Continuing...")
                    
                    self.epoch_complete = False
                
                time.sleep(0.1)
            except (EOFError, KeyboardInterrupt):
                self.should_stop = True
                break
    
    def epoch_finished(self):
        """Signal that an epoch has completed and await user decision."""
        self.epoch_complete = True
    
    def should_continue(self):
        """
        Check if training should continue.
        
        Returns:
            bool: True if training should proceed, False to stop.
        """
        return not self.should_stop

class BreedDataset(Dataset):
    """
    Optimized dataset for multi-class dog breed classification.
    
    Loads images from directory structure organized by breed names.
    Supports efficient batch loading with automatic label assignment.
    
    Attributes:
        samples: List of image file paths.
        labels: List of integer class labels.
        class_names: List of breed names.
        num_classes: Total number of breed classes.
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the breed dataset.
        
        Args:
            data_dir: Root directory containing split subdirectories.
            split: Data split to load ('train', 'val', or 'test').
            transform: Optional torchvision transforms to apply.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load paths, labels, and class mapping
        self.samples = []
        self.labels = []
        self.class_names = []
        
        split_dir = self.data_dir / split
        if split_dir.exists():
            class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            
            for class_idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                self.class_names.append(class_name)
                
                # Load images for this class
                class_samples = 0
                for img_path in class_dir.rglob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)
                        class_samples += 1
                
                if class_samples > 0:
                    print(f"   📂 {class_name}: {class_samples} images")
        
        self.num_classes = len(self.class_names)
        print(f"\n🏷️ {split.upper()}: {len(self.samples):,} samples | {self.num_classes} breeds")
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample by index.
        
        Args:
            idx: Sample index.
        
        Returns:
            tuple: (image_tensor, label) pair.
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return blank image if loading fails
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

class BreedModel(nn.Module):
    """
    Multi-class breed classification model using ResNet34 backbone.
    
    Uses ResNet34 for improved capacity when classifying 50 breeds.
    
    Attributes:
        backbone: ResNet34 feature extractor with modified FC layer.
    """
    
    def __init__(self, num_classes=50):
        """
        Initialize the breed model.
        
        Args:
            num_classes: Number of breed classes (default: 50).
        """
        super().__init__()
        # Use ResNet34 for greater capacity with 50 classes
        self.backbone = models.resnet34(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W).
        
        Returns:
            Output tensor of shape (batch, num_classes) with logits.
        """
        return self.backbone(x)

def calculate_breed_metrics(y_true, y_pred):
    """
    Calculate comprehensive multi-class classification metrics.
    
    Computes accuracy, precision, recall, F1-score for evaluating
    breed classifier performance.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
    
    Returns:
        dict: Dictionary containing all computed metrics.
    """
    metrics = {}
    
    # Standard classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Top-1 accuracy for multi-class problems
    metrics['top1_acc'] = accuracy_score(y_true, y_pred)
    
    return metrics

def calculate_topk_accuracy(outputs, targets, k=3):
    """
    Calculate Top-K accuracy for multi-class classification.
    
    Measures if the correct class appears in the top K predictions.
    
    Args:
        outputs: Model output logits.
        targets: Ground truth labels.
        k: Number of top predictions to consider.
    
    Returns:
        float: Top-K accuracy percentage.
    """
    _, pred_topk = outputs.topk(k, 1, True, True)
    pred_topk = pred_topk.t()
    correct = pred_topk.eq(targets.view(1, -1).expand_as(pred_topk))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / targets.size(0)).item()

def evaluate_breed_model(model, dataloader, device):
    """
    Evaluate breed model on a validation/test dataloader.
    
    Runs inference on all samples and computes metrics including
    Top-3 accuracy.
    
    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader with evaluation data.
        device: Torch device for inference.
    
    Returns:
        dict: Dictionary of computed metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_top3_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Top-3 accuracy
            top3_acc = calculate_topk_accuracy(outputs, labels, k=3)
            all_top3_correct += (top3_acc * labels.size(0)) / 100
            total_samples += labels.size(0)
    
    metrics = calculate_breed_metrics(all_labels, all_predictions)
    metrics['top3_acc'] = all_top3_correct / total_samples
    
    return metrics

def print_breed_header():
    """
    Print the formatted header for real-time breed training metrics.
    
    Creates a clean tabular header showing column names for
    epoch, accuracy, loss, and timing information.
    """
    print("\n" + "="*100)
    print("🐕 BREED METRICS IN REAL-TIME")
    print("="*100)
    print(f"{'EPOCH':<6} {'TRAIN ACC':<12} {'VAL ACC':<12} {'TOP-3 ACC':<12} {'LR':<12} {'LOSS':<10} {'F1':<8} {'TIME':<8}")
    print("-"*100)

def print_breed_metrics(epoch, train_acc, val_acc, top3_acc, lr, train_loss, f1, elapsed_time):
    """
    Print a single row of real-time breed training metrics.
    
    Formats and displays current epoch statistics in a consistent
    tabular format aligned with the header.
    
    Args:
        epoch: Current epoch number.
        train_acc: Training accuracy (0-1).
        val_acc: Validation accuracy (0-1).
        top3_acc: Top-3 accuracy (0-1).
        lr: Current learning rate.
        train_loss: Average training loss.
        f1: F1 score.
        elapsed_time: Epoch duration in seconds.
    """
    print(f"{epoch:<6} {train_acc*100:>9.2f}%   {val_acc*100:>9.2f}%   {top3_acc*100:>9.2f}%   {lr:>9.6f}  {train_loss:>7.4f}  {f1:>6.3f}  {elapsed_time:>6.1f}s")

def main():
    """Main training execution function for breed classification."""
    print("🐕 BREED TRAINER - REAL-TIME METRICS")
    print("🚀 50 Breeds | Train Acc | Val Acc | Top-3 Acc | Learning Rate")
    print("="*80)
    
    # Configuration optimized for 50 classes
    DATA_DIR = "breed_processed_data"
    BATCH_SIZE = 12  # Smaller batch for 50 classes on CPU
    EPOCHS = 25      # More epochs for multi-class
    LEARNING_RATE = 0.0005  # Lower LR for more classes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    print(f"🎯 Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    
    # Create directory for models
    os.makedirs("realtime_breed_models", exist_ok=True)
    
    # Transformations optimized for breeds
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\n📊 Loading breed datasets...")
    train_dataset = BreedDataset(DATA_DIR, 'train', train_transform)
    val_dataset = BreedDataset(DATA_DIR, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n🤖 Creating ResNet34 model for {train_dataset.num_classes} breeds...")
    model = BreedModel(num_classes=train_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    # Controller
    controller = RealTimeController()
    controller.start_monitoring()
    
    print("\n⚠️ CONTROL: After each epoch you can continue or stop")
    print("💡 Top-3 Accuracy: % of times correct breed is in top 3 predictions")
    
    # Print header for metrics
    print_breed_header()
    
    # Variables for tracking
    best_val_acc = 0
    best_top3_acc = 0
    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_classes': train_dataset.num_classes,
            'device': str(device)
        },
        'class_names': train_dataset.class_names,
        'epochs': []
    }
    
    for epoch in range(EPOCHS):
        if not controller.should_continue():
            print("🛑 Training stopped by user")
            break
        
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []
        
        # Progress bar with real-time info
        progress_bar = tqdm(train_loader, 
                          desc=f"Epoch {epoch+1:2d}/{EPOCHS} [50 breeds]", 
                          leave=False,
                          ncols=120)
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            if not controller.should_continue():
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with metrics
            if len(train_labels) > 0:
                current_acc = accuracy_score(train_labels, train_predictions)
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{current_acc:.3f}',
                    'LR': f'{current_lr:.6f}'
                })
        
        if not controller.should_continue():
            break
            
        # Calculate training metrics
        train_metrics = calculate_breed_metrics(train_labels, train_predictions)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_breed_model(model, val_loader, device)
        
        # Update scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Elapsed time
        elapsed_time = time.time() - start_time
        
        # Print real-time metrics
        print_breed_metrics(
            epoch + 1,
            train_metrics['accuracy'],
            val_metrics['accuracy'],
            val_metrics['top3_acc'],
            current_lr,
            avg_train_loss,
            val_metrics['f1'],
            elapsed_time
        )
        
        # Save best model
        improved = False
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            improved = True
            
        if val_metrics['top3_acc'] > best_top3_acc:
            best_top3_acc = val_metrics['top3_acc']
            improved = True
            
        if improved:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_top3_acc': best_top3_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'class_names': train_dataset.class_names
            }, f"realtime_breed_models/best_breed_model_epoch_{epoch+1}_acc_{val_metrics['accuracy']:.4f}.pth")
            print(f"    💾 Best model saved! (Val: {best_val_acc:.4f}, Top-3: {best_top3_acc:.4f})")
        
        # Save log
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': current_lr,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        training_log['epochs'].append(epoch_data)
        
        # Signal epoch complete for user control
        controller.epoch_finished()
        
        # Wait until user decides
        while controller.epoch_complete and controller.should_continue():
            time.sleep(0.1)
    
    # Finalize training
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_accuracy'] = float(best_val_acc)
    training_log['best_top3_accuracy'] = float(best_top3_acc)
    
    log_path = f"realtime_breed_models/breed_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*100)
    print(f"🎉 BREED TRAINING COMPLETED")
    print(f"🏆 Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"🥉 Best Top-3 Accuracy: {best_top3_acc:.4f} ({best_top3_acc*100:.2f}%)")
    print(f"📄 Log saved: {log_path}")
    print(f"💾 Models in: realtime_breed_models/")
    print("="*100)

if __name__ == "__main__":
    main()