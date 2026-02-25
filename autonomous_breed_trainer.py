#!/usr/bin/env python3
"""
Autonomous Dog Breed Trainer with User Control
================================================

This module implements an autonomous training system for a 50-breed
dog classification model with periodic user check-ins. Key features:

- Training with real-time metrics display (Train Acc, Val Acc, Learning Rate)
- User control every N epochs (default: 3 epochs)
- Automatic model checkpointing on improvement
- Top-3 accuracy tracking for multi-class evaluation
- Comprehensive training logs in JSON format

The system allows users to:
- Continue training for additional epochs (ENTER)
- Stop training gracefully ('q' + ENTER)
- View detailed progress summary ('s' + ENTER)

Architecture:
- ResNet34 backbone pre-trained on ImageNet
- Custom fully-connected layer for breed classification
- Optimized for 50 dog breed classes

Author: Dog Breed Classifier Team
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

class AutonomousController:
    """
    Thread-safe controller for autonomous training with user input monitoring.
    
    Manages training flow by monitoring user input in a separate thread,
    allowing graceful interruption and progress checks during training.
    
    Attributes:
        should_stop (bool): Flag indicating if training should terminate.
        check_complete (bool): Flag indicating if epoch check is pending.
        check_interval (int): Number of epochs between user check-ins.
        epochs_completed (int): Counter of completed epochs.
    """
    
    def __init__(self, check_interval=3):
        self.should_stop = False
        self.input_thread = None
        self.check_complete = False
        self.check_interval = check_interval
        self.epochs_completed = 0
    
    def start_monitoring(self):
        """
        Start the input monitoring thread.
        
        Launches a daemon thread that monitors user input in the background
        without blocking the main training loop.
        """
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
    
    def _monitor_input(self):
        """
        Monitor user input in a separate thread.
        
        Handles user commands during training pauses:
        - ENTER: Continue training for more epochs
        - 'q': Stop training gracefully
        - 's': Display detailed summary
        """
        while not self.should_stop:
            try:
                if self.check_complete:
                    print("\n" + "="*80)
                    print(f"🛑 COMPLETED {self.epochs_completed} EPOCHS - Continue?")
                    print("   ✅ ENTER = Continue 3 more epochs")
                    print("   ❌ 'q' + ENTER = Stop training")
                    print("   📊 's' + ENTER = Show detailed summary")
                    print("="*80)
                    
                    user_input = input(">>> ").strip().lower()
                    if user_input == 'q':
                        print("🛑 Stopping training...")
                        self.should_stop = True
                    elif user_input == 's':
                        print("📊 Continuing with detailed summary...")
                    else:
                        print(f"▶️ Continuing {self.check_interval} more epochs...")
                    
                    self.check_complete = False
                
                time.sleep(0.1)
            except (EOFError, KeyboardInterrupt):
                self.should_stop = True
                break
    
    def epoch_finished(self):
        """
        Notify the controller that an epoch has completed.
        
        Updates the epoch counter and sets the check flag if the
        interval threshold has been reached.
        """
        self.epochs_completed += 1
        if self.epochs_completed % self.check_interval == 0:
            self.check_complete = True
    
    def should_continue(self):
        """
        Check if training should continue.
        
        Returns:
            bool: True if training should continue, False otherwise.
        """
        return not self.should_stop
    
    def should_pause(self):
        """
        Check if training should pause for user input.
        
        Returns:
            bool: True if waiting for user input, False otherwise.
        """
        return self.check_complete

class BreedDataset(Dataset):
    """
    PyTorch Dataset optimized for 50 dog breed classification.
    
    Loads images from a directory structure where each subdirectory
    represents a breed class. Handles image loading errors gracefully.
    
    Attributes:
        data_dir (Path): Root directory containing the dataset.
        split (str): Dataset split ('train', 'val', or 'test').
        transform: PyTorch transforms to apply to images.
        samples (list): List of image file paths.
        labels (list): Corresponding class indices.
        class_names (list): List of breed class names.
        num_classes (int): Total number of breed classes.
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the breed dataset.
        
        Args:
            data_dir (str): Path to the dataset root directory.
            split (str): Which split to load ('train', 'val', 'test').
            transform: Optional PyTorch transforms to apply.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load paths, labels and class mapping
        self.samples = []
        self.labels = []
        self.class_names = []
        
        split_dir = self.data_dir / split
        if split_dir.exists():
            class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            
            for class_idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                self.class_names.append(class_name)
                
                # Load images from this class
                class_samples = 0
                for img_path in class_dir.rglob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)
                        class_samples += 1
        
        self.num_classes = len(self.class_names)
        print(f"🏷️ {split.upper()}: {len(self.samples):,} samples | {self.num_classes} breeds")
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
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
            # Return placeholder image on error to maintain batch integrity
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

class BreedModel(nn.Module):
    """
    Dog breed classification model using ResNet34 backbone.
    
    Uses transfer learning from ImageNet pretrained weights,
    replacing the final classification layer for breed prediction.
    
    Attributes:
        backbone (nn.Module): ResNet34 backbone network.
    """
    
    def __init__(self, num_classes=50):
        """
        Initialize the breed classification model.
        
        Args:
            num_classes (int): Number of breed classes. Default is 50.
        """
        super().__init__()
        self.backbone = models.resnet34(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image batch of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """
        return self.backbone(x)

def calculate_breed_metrics(y_true, y_pred):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {}
    
    # Calculate weighted metrics to handle class imbalance
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return metrics

def calculate_topk_accuracy(outputs, targets, k=3):
    """
    Calculate Top-K accuracy for multi-class classification.
    
    Args:
        outputs (torch.Tensor): Model output logits.
        targets (torch.Tensor): Ground truth labels.
        k (int): Number of top predictions to consider. Default is 3.
        
    Returns:
        float: Top-K accuracy as a percentage.
    """
    _, pred_topk = outputs.topk(k, 1, True, True)
    pred_topk = pred_topk.t()
    correct = pred_topk.eq(targets.view(1, -1).expand_as(pred_topk))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / targets.size(0)).item()

def evaluate_breed_model(model, dataloader, device):
    """
    Evaluate the breed classification model on a dataset.
    
    Performs inference on all samples in the dataloader and calculates
    comprehensive metrics including Top-K accuracy.
    
    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run evaluation on.
        
    Returns:
        dict: Dictionary containing all evaluation metrics.
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
    Print the formatted header for breed training metrics display.
    
    Displays column headers for epoch, train accuracy, validation accuracy,
    Top-3 accuracy, learning rate, loss, F1 score, and elapsed time.
    """
    print("\n" + "="*100)
    print("🐕 BREED METRICS - AUTONOMOUS TRAINING (EVERY 3 EPOCHS)")
    print("="*100)
    print(f"{'EPOCH':<6} {'TRAIN ACC':<12} {'VAL ACC':<12} {'TOP-3 ACC':<12} {'LR':<12} {'LOSS':<10} {'F1':<8} {'TIME':<8}")
    print("-"*100)

def print_breed_metrics(epoch, train_acc, val_acc, top3_acc, lr, train_loss, f1, elapsed_time):
    """
    Print formatted metrics for a single training epoch.
    
    Args:
        epoch (int): Current epoch number.
        train_acc (float): Training accuracy (0-1).
        val_acc (float): Validation accuracy (0-1).
        top3_acc (float): Top-3 accuracy (0-1).
        lr (float): Current learning rate.
        train_loss (float): Training loss value.
        f1 (float): F1 score.
        elapsed_time (float): Time taken for this epoch in seconds.
    """
    print(f"{epoch:<6} {train_acc*100:>9.2f}%   {val_acc*100:>9.2f}%   {top3_acc*100:>9.2f}%   {lr:>9.6f}  {train_loss:>7.4f}  {f1:>6.3f}  {elapsed_time:>6.1f}s")

def print_progress_summary(epochs_completed, best_val_acc, best_top3_acc, avg_time_per_epoch):
    """
    Print a progress summary at checkpoint intervals.
    
    Args:
        epochs_completed (int): Number of epochs trained so far.
        best_val_acc (float): Best validation accuracy achieved.
        best_top3_acc (float): Best Top-3 accuracy achieved.
        avg_time_per_epoch (float): Average time per epoch in seconds.
    """
    print("\n" + "🔥" * 50)
    print(f"📊 PROGRESS SUMMARY - {epochs_completed} EPOCHS COMPLETED")
    print("🔥" * 50)
    print(f"🏆 Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"🥉 Best Top-3 Accuracy: {best_top3_acc:.4f} ({best_top3_acc*100:.2f}%)")
    print(f"⏱️ Average time per epoch: {avg_time_per_epoch:.1f} seconds")
    print(f"🚀 Expected progress: {'Excellent' if best_val_acc > 0.6 else 'Very good' if best_val_acc > 0.4 else 'Normal'}")
    print("🔥" * 50)

def main():
    """
    Main entry point for autonomous breed training.
    
    Initializes the model, dataloaders, and training components,
    then runs the autonomous training loop with user checkpoints.
    """
    print("🐕 AUTONOMOUS BREED TRAINER")
    print("🚀 50 Breeds | Training every 3 epochs | Real-time metrics")
    print("="*80)
    
    # Configuration optimized for 50 classes
    DATA_DIR = "breed_processed_data"
    BATCH_SIZE = 12
    EPOCHS = 25
    LEARNING_RATE = 0.0005
    CHECK_INTERVAL = 3  # Epochs between user checkpoints
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    print(f"🎯 Total Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"⏸️ Control every: {CHECK_INTERVAL} epochs")
    
    # Create directory for models
    os.makedirs("autonomous_breed_models", exist_ok=True)
    
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
    
    # Initialize autonomous controller
    controller = AutonomousController(check_interval=CHECK_INTERVAL)
    controller.start_monitoring()
    
    print(f"\n⚠️ AUTONOMOUS CONTROL: System will train {CHECK_INTERVAL} epochs and ask to continue")
    print("💡 Top-3 Accuracy: % of times the correct breed is in top 3 predictions")
    
    # Print header for metrics
    print_breed_header()
    
    # Variables for tracking
    best_val_acc = 0
    best_top3_acc = 0
    epoch_times = []
    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'check_interval': CHECK_INTERVAL,
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
        
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []
        
        # Progress bar with breed info
        progress_bar = tqdm(train_loader, 
                          desc=f"Epoch {epoch+1:2d}/{EPOCHS} [50 breeds] - Autonomous", 
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
            
            # Update progress bar display
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
        
        # Validation phase
        val_metrics = evaluate_breed_model(model, val_loader, device)
        
        # Update scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        epoch_times.append(elapsed_time)
        
        # Display epoch metrics
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
            }, f"autonomous_breed_models/best_breed_model_epoch_{epoch+1}_acc_{val_metrics['accuracy']:.4f}.pth")
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
        
        # Notify controller that epoch finished
        controller.epoch_finished()
        
        # Show summary at checkpoint intervals
        if (epoch + 1) % CHECK_INTERVAL == 0:
            avg_time = sum(epoch_times[-CHECK_INTERVAL:]) / CHECK_INTERVAL
            print_progress_summary(epoch + 1, best_val_acc, best_top3_acc, avg_time)
        
        # Wait for user input at checkpoints
        while controller.should_pause() and controller.should_continue():
            time.sleep(0.1)
    
    # Finalize training
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_accuracy'] = float(best_val_acc)
    training_log['best_top3_accuracy'] = float(best_top3_acc)
    
    log_path = f"autonomous_breed_models/autonomous_breed_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*100)
    print(f"🎉 AUTONOMOUS BREED TRAINING COMPLETED")
    print(f"🏆 Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"🥉 Best Top-3 Accuracy: {best_top3_acc:.4f} ({best_top3_acc*100:.2f}%)")
    print(f"⏱️ Average time per epoch: {sum(epoch_times)/len(epoch_times):.1f}s")
    print(f"📄 Log saved: {log_path}")
    print(f"💾 Models in: autonomous_breed_models/")
    print("="*100)

if __name__ == "__main__":
    main()