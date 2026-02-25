#!/usr/bin/env python3
"""
Balanced Dataset Training Module.

Retrains the breed classification model using a balanced dataset where
each breed class has the same number of training images, eliminating
class imbalance bias.

Features:
    - BalancedBreedDataset for uniform class distribution
    - ImprovedBreedClassifier with ResNet50 backbone
    - Advanced data augmentation pipeline
    - Training/validation split with reproducible seeding
    - Automatic checkpointing of best models
    - Training metrics visualization

Usage:
    Requires balanced dataset from balance_dataset.py
    python train_balanced_model.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class BalancedBreedDataset(Dataset):
    """
    Dataset for balanced breed classification training.
    
    Loads images from a directory structure where each breed is a
    subdirectory containing balanced numbers of samples.
    
    Attributes:
        data_dir (str): Path to the data directory.
        transform: Image transformation pipeline.
        samples (list): List of (image_path, class_idx) tuples.
        class_to_idx (dict): Mapping of breed names to class indices.
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the balanced dataset.
        
        Args:
            data_dir: Path to directory containing breed subdirectories.
            transform: Optional transform to apply to images.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        
        # Get all breeds and create mapping
        breeds = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
        
        for idx, breed in enumerate(breeds):
            self.class_to_idx[breed] = idx
            breed_path = os.path.join(data_dir, breed)
            
            for img_file in os.listdir(breed_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(breed_path, img_file), idx))
        
        print(f" Balanced dataset loaded:")
        print(f"   Total images: {len(self.samples)}")
        print(f"   Total classes: {len(self.class_to_idx)}")
        print(f"   Average per class: {len(self.samples) / len(self.class_to_idx):.1f}")
        
        # Verify balance
        class_counts = {}
        for _, class_idx in self.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        counts = list(class_counts.values())
        print(f"   Min/Max per class: {min(counts)}/{max(counts)}")
        print(f"   Standard deviation: {np.std(counts):.2f}")
        
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx (int): Sample index.
            
        Returns:
            tuple: (image_tensor, label) pair.
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))


class ImprovedBreedClassifier(nn.Module):
    """
    Improved breed classifier with ResNet50 backbone.
    
    Features partial layer freezing for transfer learning and
    a multi-layer classification head with dropout.
    
    Attributes:
        backbone (nn.Module): ResNet50 with custom final layers.
    """
    
    def __init__(self, num_classes: int = 50):
        """
        Initialize the improved classifier.
        
        Args:
            num_classes (int): Number of breed classes. Default: 50.
        """
        super().__init__()
        # Use ResNet50 for better performance
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        # Replace final classifier
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)

def train_balanced_model():
    """
    Train the breed classifier using a balanced dataset.
    
    Trains the improved ResNet50 model with data augmentation, learning
    rate scheduling, and automatic checkpointing of the best model.
    
    Returns:
        tuple: (best_val_accuracy, class_to_idx mapping)
    """
    
    print(" TRAINING WITH BALANCED DATASET")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")
    
    num_classes = 50
    batch_size = 16
    num_epochs = 25
    learning_rate = 0.001
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
    
    # Full dataset
    full_dataset = BalancedBreedDataset(
        'breed_processed_data/train',
        transform=None  # Apply transform later
    )
    
    # Train/validation split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms after split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    print(f" Data split:")
    print(f"   Training: {len(train_dataset):,} images")
    print(f"   Validation: {len(val_dataset):,} images")
    print(f"   Validation: {len(val_dataset):,} images")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Model
    model = ImprovedBreedClassifier(num_classes).to(device)
    
    # Loss and optimizer (no class weights needed with balanced dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=3, 
        factor=0.5
    )
    
    # Tracking variables
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print(f"\n Starting training ({num_epochs} epochs)...")
    
    for epoch in range(num_epochs):
        # === training ===
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_preds += target.size(0)
            correct_preds += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_preds / total_preds
        
        # === Validation ===
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        # Update tracking
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train: Loss {epoch_train_loss:.4f}, Acc {epoch_train_acc:.2f}%")
        print(f"  Val:   Loss {epoch_val_loss:.4f}, Acc {epoch_val_acc:.2f}%")
        
        # Scheduler step
        scheduler.step(epoch_val_acc)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            
            # Create directory for balanced models
            os.makedirs('balanced_models', exist_ok=True)
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accuracy': epoch_train_acc,
                'val_accuracy': epoch_val_acc,
                'class_to_idx': full_dataset.class_to_idx,
                'model_architecture': 'ResNet50',
                'dataset_balanced': True,
                'images_per_class': 161
            }, f'balanced_models/best_balanced_breed_model_epoch_{epoch+1}_acc_{epoch_val_acc:.4f}.pth')
            
            print(f" New best model saved: {epoch_val_acc:.2f}%")
    
    print(f"\n Best validation accuracy: {best_val_acc:.2f}%")
    
    # Create training plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy During Training')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('balanced_training_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save training metrics
    metrics = {
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accuracies[-1],
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'num_epochs': num_epochs,
        'dataset_info': {
            'balanced': True,
            'images_per_class': 161,
            'total_classes': num_classes,
            'total_images': len(full_dataset)
        }
    }
    
    with open('balanced_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Results saved:")
    print(f"    Metrics: balanced_training_metrics.json")
    print(f"    Plots: balanced_training_metrics.png")
    print(f"    Model: balanced_models/best_balanced_breed_model_epoch_*")
    
    return best_val_acc, full_dataset.class_to_idx


if __name__ == "__main__":
    # Verify balanced dataset exists
    if not os.path.exists('balancing_final_report.json'):
        print(" First run balance_dataset.py")
        exit(1)
    
    # Train model
    best_acc, class_mapping = train_balanced_model()
    
    print(f"\n TRAINING COMPLETED")
    print(f" Best accuracy: {best_acc:.2f}%")
    print(f" Perfectly balanced dataset used")
    print(f" Ready to integrate the new model")