"""
Simplified and Stable Binary Dog Classifier Trainer.

Provides a streamlined training pipeline for binary dog/no-dog classification
using ResNet18 backbone with stable PyTorch operations.

Features:
    - ResNet18 backbone for efficiency
    - Manual stop control during training
    - Real-time progress tracking with tqdm
    - Automatic model checkpointing
    - Optimized for CPU training
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure environment before importing PyTorch
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

# Simplified stop control
class SimpleController:
    """Simple training controller with keyboard stop detection.
    
    Allows users to stop training gracefully by pressing 'q' during execution.
    
    Attributes:
        should_stop: Boolean flag indicating stop request.
    """
    
    def __init__(self):
        """Initialize the controller."""
        self.should_stop = False
    
    def check_for_stop(self):
        """Check if user requested training stop via keyboard input.
        
        Returns:
            bool: True if stop requested, False otherwise.
        """
        try:
            import select
            import sys
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                input_line = sys.stdin.readline()
                if 'q' in input_line.lower():
                    self.should_stop = True
                    return True
        except:
            pass
        return False

class SimpleBinaryDataset(Dataset):
    """Simplified binary classification dataset.
    
    Loads dog and non-dog images from structured directories with
    support for nested subdirectories.
    
    Attributes:
        data_path: Root path to the dataset.
        transform: Image transformations to apply.
        samples: List of (image_path, label) tuples.
    """
    
    def __init__(self, data_path, transform=None, max_per_class=10000):
        """Initialize the binary dataset.
        
        Args:
            data_path: Path to root data directory.
            transform: Optional torchvision transforms.
            max_per_class: Maximum samples per class.
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        self._load_samples(max_per_class)
        
    def _load_samples(self, max_per_class):
        """Load image samples from disk.
        
        Args:
            max_per_class: Maximum number of samples per class.
        """
        print(" Loading simplified binary dataset...")
        
        # NODOG class
        nodog_path = self.data_path / "NODOG"
        no_dog_count = 0
        if nodog_path.exists():
            # Load direct files
            for img_file in nodog_path.glob("*.jpg"):
                if no_dog_count >= max_per_class:
                    break
                self.samples.append((str(img_file), 0))
                no_dog_count += 1
            
            # Load from subdirectories
            for subdir in nodog_path.iterdir():
                if subdir.is_dir() and no_dog_count < max_per_class:
                    for img_file in subdir.glob("*.jpg"):
                        if no_dog_count >= max_per_class:
                            break
                        self.samples.append((str(img_file), 0))
                        no_dog_count += 1
        
        print(f"    NO-DOG: {no_dog_count:,} images")
        
        # DOG class
        yesdog_path = self.data_path / "YESDOG"
        dog_count = 0
        if yesdog_path.exists():
            for breed_dir in yesdog_path.iterdir():
                if breed_dir.is_dir() and dog_count < max_per_class:
                    for img_file in list(breed_dir.glob("*.JPEG")) + list(breed_dir.glob("*.jpg")):
                        if dog_count >= max_per_class:
                            break
                        self.samples.append((str(img_file), 1))
                        dog_count += 1
        
        print(f"    DOG: {dog_count:,} images")
        print(f"    Total: {len(self.samples):,} samples")
        
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
            # If a image fails, return a blank image
            print(f" Error loading {img_path}: {e}")
            blank_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, label

class SimpleBinaryModel(nn.Module):
    """Simplified binary classifier using ResNet18.
    
    Lightweight model for dog/no-dog classification using pretrained
    ResNet18 backbone.
    
    Attributes:
        backbone: ResNet18 feature extractor with modified classifier.
    """
    
    def __init__(self):
        """Initialize with pretrained ResNet18."""
        super().__init__()
        # Use ResNet18 instead of EfficientNet to avoid issues
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 2)  # 2 classes: dog/no-dog
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, 3, H, W).
            
        Returns:
            Output logits of shape (N, 2).
        """
        return self.backbone(x)

class SimpleTrainer:
    """Simplified training manager for binary classification.
    
    Handles training loop, validation, checkpointing, and stop control.
    
    Attributes:
        model: The neural network model.
        device: Computation device (CPU/GPU).
        controller: Stop controller for graceful termination.
        optimizer: Adam optimizer.
        criterion: CrossEntropyLoss function.
    """
    
    def __init__(self, model, device='cpu'):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device for computation.
        """
        self.model = model.to(device)
        self.device = device
        self.controller = SimpleController()
        
        # Simple configuration
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
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
            # Check for stop request every 10 batches
            if batch_idx % 10 == 0 and self.controller.check_for_stop():
                print("\n Stop requested by user")
                break
                
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
            
            # Update bar
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
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train_model(self, train_loader, val_loader, epochs=20, save_path='./binary_models'):
        """Complete training loop with validation and checkpointing.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of training epochs.
            save_path: Directory to save model checkpoints.
            
        Returns:
            dict: Training results including best accuracy.
        """
        print(" SIMPLIFIED BINARY TRAINING")
        print("=" * 60)
        print(f" Epochs: {epochs}")
        print(f" Device: {self.device}")
        print("  Press Enter + 'q' + Enter to stop")
        print()
        
        Path(save_path).mkdir(exist_ok=True)
        best_val_acc = 0
        
        for epoch in range(1, epochs + 1):
            if self.controller.should_stop:
                print(f" Training stopped at epoch {epoch}")
                break
                
            print(f" EPOCH {epoch}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Show results
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = Path(save_path) / 'best_binary_model.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': val_acc,
                    'epoch': epoch,
                }, model_path)
                print(f" Best model saved: {val_acc:.2f}%")
            
            print()
        
        print(f" BEST ACCURACY ACHIEVED: {best_val_acc:.2f}%")
        return {'best_accuracy': best_val_acc}

def get_simple_transforms():
    """Get simplified image transformation pipelines.
    
    Returns:
        tuple: (train_transform, val_transform) pipelines.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
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
    """Main entry point for binary classifier training."""
    print(" SIMPLIFIED BINARY TRAINER")
    print(" Stable version without PyTorch conflicts")
    print("=" * 80)
    
    # Configuration
    DATA_PATH = "./DATASETS"
    BATCH_SIZE = 16  # Conservative batch size
    EPOCHS = 20
    MAX_PER_CLASS = 10000  # 10k per class
    
    # Verify data
    if not Path(DATA_PATH).exists():
        print(f" Data directory not found: {DATA_PATH}")
        return
    
    # Transformations
    train_transform, val_transform = get_simple_transforms()
    
    # Datasets
    print(" Creating datasets...")
    train_dataset = SimpleBinaryDataset(DATA_PATH, train_transform, MAX_PER_CLASS)
    val_dataset = SimpleBinaryDataset(DATA_PATH, val_transform, MAX_PER_CLASS//3)
    
    # DataLoaders simplificados
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f" Train samples: {len(train_dataset):,}")
    print(f" Val samples: {len(val_dataset):,}")
    print()
    
    # Model
    print(" Creating ResNet18 model...")
    model = SimpleBinaryModel()
    device = torch.device('cpu')
    
    # Trainer
    trainer = SimpleTrainer(model, device)
    
    # Train
    results = trainer.train_model(train_loader, val_loader, EPOCHS)
    
    print(" TRAINING COMPLETED!")
    print(f" Best accuracy: {results['best_accuracy']:.2f}%")
    print(f" Model saved at: binary_models/best_binary_model.pth")
    
    # Copy best model to expected location
    import shutil
    src = "binary_models/best_binary_model.pth"
    dst = "best_model.pth"
    if Path(src).exists():
        shutil.copy2(src, dst)
        print(f" Model copied to: {dst}")
    
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