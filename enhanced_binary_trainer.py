#!/usr/bin/env python3
"""
Enhanced Binary Trainer Module
==============================

This module implements an advanced binary classifier trainer for dog detection.
It provides comprehensive training with per-epoch control, detailed metrics tracking,
and interactive training management capabilities.

Features:
    - Interactive epoch-by-epoch training control
    - Comprehensive metrics calculation (accuracy, precision, recall, F1, AUC)
    - Confusion matrix analysis with specificity and sensitivity
    - Model checkpointing with automatic best model saving
    - Training log generation in JSON format
    - Data augmentation pipeline for robust training

Architecture:
    - Backbone: ResNet18 (pretrained on ImageNet)
    - Binary classification: dog vs. not-dog

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

class EnhancedController:
    """
    Interactive training controller for epoch-by-epoch management.
    
    This class provides a mechanism for users to control training flow,
    allowing them to continue or stop training after each epoch completes.
    It runs input monitoring in a separate daemon thread.
    
    Attributes:
        should_stop (bool): Flag indicating if training should stop.
        input_thread (threading.Thread): Background thread for input monitoring.
        epoch_complete (bool): Flag indicating if the current epoch has finished.
    """
    
    def __init__(self):
        self.should_stop = False
        self.input_thread = None
        self.epoch_complete = False
    
    def start_monitoring(self):
        """
        Start the input monitoring thread.
        
        Spawns a daemon thread that monitors user input for training control.
        """
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
    
    def _monitor_input(self):
        """
        Monitor user input in a separate thread.
        
        Listens for user commands after each epoch to determine whether
        to continue or stop training.
        """
        while not self.should_stop:
            try:
                if self.epoch_complete:
                    print("\n" + "="*80)
                    print("🛑 EPOCH COMPLETED - Continue training?")
                    print("   ✅ Press ENTER to continue")
                    print("   ❌ Type 'q' + ENTER to stop")
                    print("="*80)
                    
                    user_input = input(">>> ").strip().lower()
                    if user_input == 'q':
                        print("🛑 Stopping training by user request...")
                        self.should_stop = True
                    else:
                        print("▶️ Continuing with the next epoch...")
                    
                    self.epoch_complete = False
                
                time.sleep(0.1)
            except (EOFError, KeyboardInterrupt):
                self.should_stop = True
                break
    
    def epoch_finished(self):
        """
        Signal that the current epoch has completed.
        
        Sets the epoch_complete flag to True, which triggers the
        user prompt in the monitoring thread.
        """
        self.epoch_complete = True
    
    def should_continue(self):
        """
        Check if training should continue.
        
        Returns:
            bool: True if training should continue, False otherwise.
        """
        return not self.should_stop

class EnhancedBinaryDataset(Dataset):
    """
    PyTorch Dataset for binary dog classification.
    
    Loads images from a directory structure with 'dog' and 'nodog' subdirectories
    and provides them for training/validation with optional transformations.
    
    Args:
        data_dir (str or Path): Root directory containing the dataset.
        split (str): Data split to load ('train', 'val', or 'test').
        transform (callable, optional): Transform to apply to images.
    
    Attributes:
        samples (list): List of image file paths.
        labels (list): List of corresponding labels (0 for not-dog, 1 for dog).
    
    Directory Structure:
        data_dir/
            train/
                nodog/
                dog/
            val/
                nodog/
                dog/
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load image paths and corresponding labels
        self.samples = []
        self.labels = []
        
        # Class 0: NO-dog (nodog)
        no_dog_dir = self.data_dir / split / "nodog"
        if no_dog_dir.exists():
            for img_path in no_dog_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append(str(img_path))
                    self.labels.append(0)
        
        # Class 1: dog (positive class)
        dog_dir = self.data_dir / split / "dog"
        if dog_dir.exists():
            for img_path in dog_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append(str(img_path))
                    self.labels.append(1)
        
        print(f"📊 Dataset {split}: {len(self.samples)} samples")
        print(f"   ❌ NOT-DOG: {sum(1 for l in self.labels if l == 0):,}")
        print(f"   ✅ DOG: {sum(1 for l in self.labels if l == 1):,}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image_tensor, label) where image_tensor is the transformed
                   image and label is 0 (not-dog) or 1 (dog).
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return a blank image as fallback for corrupted files
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

class EnhancedBinaryModel(nn.Module):
    """
    Enhanced binary classification model based on ResNet18.
    
    Uses a pretrained ResNet18 backbone with a modified final fully connected
    layer for binary dog/not-dog classification.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary).
    
    Architecture:
        - Backbone: ResNet18 (pretrained on ImageNet)
        - Output: Linear layer with num_classes outputs
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)

def calculate_metrics(y_true, y_pred, y_scores):
    """
    Calculate comprehensive classification metrics.
    
    Computes accuracy, precision, recall, F1-score, AUC-ROC, and confusion
    matrix statistics including specificity and sensitivity.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_scores (np.ndarray): Prediction scores/probabilities of shape (n_samples, n_classes).
    
    Returns:
        dict: Dictionary containing all computed metrics:
            - accuracy: Overall classification accuracy
            - precision: Weighted precision score
            - recall: Weighted recall score
            - f1: Weighted F1-score
            - auc: Area Under ROC Curve (for binary classification)
            - confusion_matrix: Confusion matrix array
            - true_negatives, false_positives, false_negatives, true_positives
            - specificity: True Negative Rate
            - sensitivity: True Positive Rate (same as recall for positive class)
    """
    metrics = {}
    
    # Calculate basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC-ROC (only for binary classification)
    if len(np.unique(y_true)) == 2:
        metrics['auc'] = roc_auc_score(y_true, y_scores[:, 1])
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Calculate specificity (TNR) and sensitivity (TPR)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

def print_metrics(metrics, split_name=""):
    """
    Print formatted classification metrics to console.
    
    Displays accuracy, precision, recall, F1-score, AUC (if available),
    specificity, sensitivity, and confusion matrix in a readable format.
    
    Args:
        metrics (dict): Dictionary of metrics from calculate_metrics().
        split_name (str): Name of the data split (e.g., 'train', 'validation').
    """
    print(f"\n📊 {split_name.upper()} METRICS")
    print("="*60)
    print(f"🎯 Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"🎯 Precision:  {metrics['precision']:.4f}")
    print(f"🎯 Recall:     {metrics['recall']:.4f}")
    print(f"🎯 F1-Score:   {metrics['f1']:.4f}")
    
    if 'auc' in metrics:
        print(f"📈 AUC:        {metrics['auc']:.4f}")
    
    if 'specificity' in metrics and 'sensitivity' in metrics:
        print(f"🔍 Specificity: {metrics['specificity']:.4f}")
        print(f"🔍 Sensitivity: {metrics['sensitivity']:.4f}")
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"    Pred:  [NOT-DOG] [DOG]")
        print(f"Actual NOT-DOG:  {cm[0,0]:6d}   {cm[0,1]:6d}")
        print(f"Actual DOG:      {cm[1,0]:6d}   {cm[1,1]:6d}")

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset and compute metrics.
    
    Performs inference on the entire dataloader and calculates comprehensive
    classification metrics.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run evaluation on.
    
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    return calculate_metrics(all_labels, all_predictions, np.array(all_scores))

def save_training_log(log_data, log_path):
    """
    Save training log to a JSON file.
    
    Args:
        log_data (dict): Dictionary containing training history and metrics.
        log_path (str): Path to save the JSON log file.
    """
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)

def main():
    """
    Main function to run the enhanced binary training pipeline.
    
    Initializes datasets, model, optimizer, and training loop with
    interactive epoch control and comprehensive metrics tracking.
    """
    print("🐕 ENHANCED BINARY TRAINER")
    print("🚀 With comprehensive metrics and per-epoch control")
    print("="*80)
    
    # Configuration
    DATA_DIR = "processed_data"
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # Create directory for saving models
    os.makedirs("enhanced_binary_models", exist_ok=True)
    
    # Optimized data augmentation transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = EnhancedBinaryDataset(DATA_DIR, 'train', train_transform)
    val_dataset = EnhancedBinaryDataset(DATA_DIR, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("🤖 Creating ResNet18 model...")
    model = EnhancedBinaryModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Enhanced controller for interactive training
    controller = EnhancedController()
    controller.start_monitoring()
    
    print("\n🚀 ENHANCED BINARY TRAINING")
    print("="*80)
    print(f"🎯 Epochs: {EPOCHS}")
    print(f"🔄 Batch Size: {BATCH_SIZE}")
    print(f"📚 Learning Rate: {LEARNING_RATE}")
    print(f"💻 Device: {device}")
    print("⚠️ The system will ask after each epoch whether to continue")
    
    # Variables for tracking
    best_val_acc = 0
    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'device': str(device)
        },
        'epochs': []
    }
    
    for epoch in range(EPOCHS):
        if not controller.should_continue():
            print("🛑 Training stopped by user")
            break
            
        print(f"\n📅 EPOCH {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []
        train_scores = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
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
            
            # Accumulate predictions for metrics calculation
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_scores.extend(scores.detach().cpu().numpy())
            
            # Update progress bar with current batch accuracy
            current_acc = accuracy_score(train_labels[-len(labels):], 
                                       train_predictions[-len(labels):])
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        if not controller.should_continue():
            break
            
        # Calculate training metrics for the epoch
        train_metrics = calculate_metrics(train_labels, train_predictions, np.array(train_scores))
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        print("\n🔍 Evaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Print epoch results
        print(f"\n🏃 EPOCH {epoch + 1} RESULTS")
        print("="*60)
        print(f"📉 Train Loss: {avg_train_loss:.4f}")
        print_metrics(train_metrics, "TRAIN")
        print_metrics(val_metrics, "VALIDATION")
        
        # Save best model checkpoint
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, f"enhanced_binary_models/best_model_epoch_{epoch+1}_acc_{val_metrics['accuracy']:.4f}.pth")
            print(f"💾 Best model saved! (Acc: {best_val_acc:.4f})")
        
        # Save log
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': scheduler.get_last_lr()[0],
            'timestamp': datetime.now().isoformat()
        }
        training_log['epochs'].append(epoch_data)
        
        # Signal epoch completion for user interaction
        controller.epoch_finished()
        
        # Wait for user decision on whether to continue
        while controller.epoch_complete and controller.should_continue():
            time.sleep(0.1)
    
    # Finalize training and save log
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_accuracy'] = float(best_val_acc)
    
    log_path = f"enhanced_binary_models/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_training_log(training_log, log_path)
    
    print(f"\n🎉 TRAINING COMPLETED")
    print("="*60)
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"📄 Log saved to: {log_path}")
    print("✅ All models saved to: enhanced_binary_models/")

if __name__ == "__main__":
    main()