"""
Simplified and Stable Dog Breed Classifier Trainer.

Provides a streamlined training pipeline for 50-breed dog classification
using ResNet34 backbone with stable PyTorch operations.

Features:
    - ResNet34 backbone for better accuracy
    - Learning rate scheduling with StepLR
    - Early stopping with patience
    - Automatic model checkpointing
    - Optimized for CPU training
"""

import os
import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure environment
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

class SimpleBreedDataset(Dataset):
    """Simplified breed classification dataset.
    
    Loads dog images from structured directories organized by breed
    with support for train/val splits.
    
    Attributes:
        data_path: Root path to the processed dataset.
        transform: Image transformations to apply.
        samples: List of (image_path, label) tuples.
        num_classes: Total number of breed classes.
        info: Dataset metadata from JSON configuration.
    """
    
    def __init__(self, data_path="./breed_processed_data", split="train", transform=None):
        """Initialize the breed dataset.
        
        Args:
            data_path: Path to processed breed data directory.
            split: Dataset split to load ('train' or 'val').
            transform: Optional torchvision transforms.
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        
        # Load configuration
        with open(self.data_path / "dataset_info.json", 'r') as f:
            self.info = json.load(f)
        
        self.num_classes = self.info['total_breeds']
        self._load_split_samples(split)
        
    def _load_split_samples(self, split):
        """Load samples from the specified data split.
        
        Args:
            split: Split name ('train' or 'val').
        """
        print(f" Loading {split} split...")
        
        split_dir = self.data_path / split
        if not split_dir.exists():
            print(f" Directory not found: {split_dir}")
            return
        
        # Load samples from each class folder
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Find the class index
                class_idx = None
                for breed, info in self.info['breed_details'].items():
                    if info['display_name'].lower().replace(' ', '_') == class_name.lower():
                        class_idx = info['class_index']
                        break
                
                if class_idx is None:
                    # Search by direct name
                    if class_name in self.info['breed_details']:
                        class_idx = self.info['breed_details'][class_name]['class_index']
                    else:
                        print(f" Class not found: {class_name}")
                        continue
                
                # Load images from this class
                for img_file in class_dir.glob("*.JPEG"):
                    self.samples.append((str(img_file), class_idx))
                for img_file in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_file), class_idx))
        
        print(f"    {len(self.samples):,} samples loaded")
        
    def __len__(self):
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample by index.
        
        Args:
            idx: Sample index.
            
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
            print(f" Error loading {img_path}: {e}")
            # Return blank image if load fails
            blank_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, label

class SimpleBreedModel(nn.Module):
    """Simplified breed classifier using ResNet34.
    
    Uses pretrained ResNet34 backbone for multi-class breed classification.
    
    Attributes:
        backbone: ResNet34 feature extractor with modified classifier.
    """
    
    def __init__(self, num_classes=50):
        """Initialize with pretrained ResNet34.
        
        Args:
            num_classes: Number of breed classes (default: 50).
        """
        super().__init__()
        # Use ResNet34 for better accuracy with reasonable speed
        self.backbone = models.resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, 3, H, W).
            
        Returns:
            Output logits of shape (N, num_classes).
        """
        return self.backbone(x)

class SimpleBreedTrainer:
    """Simplified training manager for breed classification.
    
    Handles training loop, validation, learning rate scheduling,
    early stopping, and model checkpointing.
    
    Attributes:
        model: The neural network model.
        device: Computation device (CPU/GPU).
        optimizer: Adam optimizer with weight decay.
        criterion: CrossEntropyLoss function.
        scheduler: StepLR learning rate scheduler.
    """
    
    def __init__(self, model, device='cpu'):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device for computation.
        """
        self.model = model.to(device)
        self.device = device
        
        # Simple but effective configuration
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.
            
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update bar every 10 batches
            if batch_idx % 10 == 0:
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model on validation set.
        
        Args:
            val_loader: DataLoader for validation data.
            
        Returns:
            tuple: (validation_loss, validation_accuracy)
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train_model(self, train_loader, val_loader, epochs=25, save_path='./breed_models'):
        """Complete training loop with validation, scheduling, and early stopping.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of training epochs.
            save_path: Directory to save model checkpoints.
            
        Returns:
            dict: Training results including best accuracy and final epoch.
        """
        print(" SIMPLIFIED BREED TRAINING")
        print("=" * 60)
        print(f" Epochs: {epochs}")
        print(f" Classes: {self.model.backbone.fc.out_features}")
        print(f" Device: {self.device}")
        print()
        
        Path(save_path).mkdir(exist_ok=True)
        best_val_acc = 0
        patience_counter = 0
        patience = 5
        
        for epoch in range(1, epochs + 1):
            print(f" EPOCH {epoch}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Show results
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f" Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                model_path = Path(save_path) / 'best_breed_model_simple.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': val_acc,
                    'epoch': epoch,
                    'num_classes': self.model.backbone.fc.out_features
                }, model_path)
                print(f" Best model saved: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f" Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f" Early stopping at epoch {epoch}")
                break
            
            print()
        
        print(f" BEST ACCURACY ACHIEVED: {best_val_acc:.2f}%")
        return {'best_accuracy': best_val_acc, 'final_epoch': epoch}

def get_breed_transforms():
    """Get image transformation pipelines for breed classification.
    
    Returns:
        tuple: (train_transform, val_transform) pipelines.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    """Main entry point for breed classifier training."""
    print(" SIMPLIFIED BREED TRAINER")
    print(" 50 Breeds - Stable version")
    print("=" * 80)
    
    # Configuration
    DATA_PATH = "./breed_processed_data"
    BATCH_SIZE = 16  # Conservative for 50 classes
    EPOCHS = 25
    
    # Verify processed data exists
    if not Path(DATA_PATH).exists():
        print(f" Processed data not found: {DATA_PATH}")
        print(" Run first: python breed_preprocessor.py")
        return
    
    # Transformations
    train_transform, val_transform = get_breed_transforms()
    
    # Datasets
    print(" Creating breed datasets...")
    train_dataset = SimpleBreedDataset(DATA_PATH, "train", train_transform)
    val_dataset = SimpleBreedDataset(DATA_PATH, "val", val_transform)
    
    print(f" Train samples: {len(train_dataset):,}")
    print(f" Val samples: {len(val_dataset):,}")
    print(f" Number of breeds: {train_dataset.num_classes}")
    print()
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model
    print(" Creating ResNet34 model for 50 breeds...")
    model = SimpleBreedModel(num_classes=train_dataset.num_classes)
    device = torch.device('cpu')
    
    # Trainer
    trainer = SimpleBreedTrainer(model, device)
    
    # Train
    results = trainer.train_model(train_loader, val_loader, EPOCHS)
    
    print(" BREED TRAINING COMPLETED!")
    print(f" Best accuracy: {results['best_accuracy']:.2f}%")
    print(f" Epochs trained: {results['final_epoch']}")
    print(f" Model saved at: breed_models/best_breed_model_simple.pth")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()