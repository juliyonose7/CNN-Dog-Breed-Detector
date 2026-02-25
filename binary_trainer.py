"""
Binary Dog Classifier Trainer
=============================

A training module for binary classification of images into DOG vs NO-DOG categories.
Optimized for AMD 7800X3D CPU with multi-threaded data loading and efficient batch processing.

Features:
    - EfficientNet-B1 backbone with custom classifier head
    - OneCycleLR learning rate scheduling
    - Manual training control (press 'q' to stop safely)
    - Balanced dataset loading with configurable sample limits
    - Comprehensive training reports with confusion matrices
    - Early stopping with patience control
    - Gradient clipping for stable training

Usage:
    Run directly: python binary_trainer.py
    
Optimized for: AMD 7800X3D (16 threads)
"""

# === IMPORTS ===

# Standard library
import os
import sys
import time
import json
import threading

# Deep learning framework
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# Visualization and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import warnings
warnings.filterwarnings('ignore')

# === PLATFORM-SPECIFIC IMPORTS ===
if sys.platform == "win32":
    import msvcrt
else:
    import select

# Control of parada manual
class TrainingController:
    """Controller for manual training interruption and status monitoring.
    
    Provides safe training interruption by monitoring keyboard input
    in a separate thread. Supports both Windows and Unix platforms.
    
    Attributes:
        should_stop: Flag indicating if training should stop.
        input_thread: Background thread monitoring user input.
        monitoring: Flag indicating if input monitoring is active.
    """
    
    def __init__(self):
        """Initialize the training controller with default state."""
        self.should_stop = False
        self.input_thread = None
        self.monitoring = False
    
    def start_monitoring(self):
        """Start monitoring user input for training control commands.
        
        Starts a daemon thread that listens for:
            - 'q': Stop training safely after current epoch
            - 's': Show current training status
        """
        self.monitoring = True
        self.should_stop = False
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
        print("🔄 TRAINING CONTROL ACTIVATED")
        print("   Press 'q' + Enter to stop training safely")
        print("   Press 's' + Enter to show statistics")
        print("=" * 70)
    
    def _monitor_input(self):
        """Monitor user input in a separate thread.
        
        Platform-aware implementation that handles keyboard input
        on both Windows (msvcrt) and Unix (select) systems.
        """
        while self.monitoring:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        if key == 'q':
                            print("\n🛑 STOP REQUESTED - Finishing current epoch safely...")
                            self.should_stop = True
                            break
                        elif key == 's':
                            print(f"\n📊 STATUS: Training in progress... (Press 'q' to stop)")
                else:
                    # For Unix/Linux systems
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.readline().strip().lower()
                        if key == 'q':
                            print("\n🛑 STOP REQUESTED - Finishing current epoch safely...")
                            self.should_stop = True
                            break
                        elif key == 's':
                            print(f"\n📊 STATUS: Training in progress... (Press 'q' to stop)")
                
                time.sleep(0.1)
            except:
                time.sleep(0.1)
                continue
    
    def stop_monitoring(self):
        """Stop the input monitoring thread.
        
        Waits up to 1 second for the monitoring thread to terminate gracefully.
        """
        self.monitoring = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)


# ===================================================================
# AMD 7800X3D CONFIGURATION
# ===================================================================


def optimize_for_7800x3d():
    """Configure environment variables and PyTorch settings for AMD 7800X3D.
    
    Sets up optimal thread counts for various numerical libraries to take
    full advantage of the 16-thread AMD 7800X3D processor.
    
    Configures:
        - MKL, NumExpr, OpenMP, OpenBLAS threads: 16
        - PyTorch intra-op threads: 16
        - PyTorch inter-op threads: 4
    """
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '16'
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['OPENBLAS_NUM_THREADS'] = '16'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '16'
    
    # Configure PyTorch threading
    torch.set_num_threads(16)
    torch.set_num_interop_threads(4)
    
    print("🚀 7800X3D environment variables configured")
    print(f"💻 CPU threads: {torch.get_num_threads()}")

# ===================================================================
# BINARY DATASET
# ===================================================================


class BinaryDogDataset(Dataset):
    """PyTorch Dataset for binary dog classification.
    
    Loads and manages images for binary classification (dog vs no-dog).
    Supports automatic class balancing and configurable sample limits.
    
    Args:
        data_path: Path to the dataset root directory containing YESDOG and NODOG folders.
        transform: Optional torchvision transforms to apply to images.
        max_samples_per_class: Optional limit on samples per class for balancing.
    
    Attributes:
        samples: List of (image_path, label) tuples.
        classes: Class names ['no_dog', 'dog'] mapping to labels [0, 1].
    """
    
    def __init__(self, data_path, transform=None, max_samples_per_class=None):
        """Initialize the dataset with path and optional transforms."""
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        self.classes = ['no_dog', 'dog']  # 0: no-dog, 1: dog
        
        self._load_samples(max_samples_per_class)
        
    def _load_samples(self, max_samples_per_class):
        """Load samples from the dataset directory.
        
        Args:
            max_samples_per_class: Maximum samples to load per class (None for unlimited).
        """
        print("🔄 Loading binary dataset...")
        
        # Load NO-DOG images
        nodog_path = self.data_path / "NODOG"
        if nodog_path.exists():
            nodog_count = 0
            
            # Individual files in root
            for img_file in nodog_path.glob("*.jpg"):
                if max_samples_per_class and nodog_count >= max_samples_per_class:
                    break
                self.samples.append((str(img_file), 0))
                nodog_count += 1
                
            # Files in subdirectories
            for subdir in nodog_path.iterdir():
                if subdir.is_dir():
                    for img_file in subdir.glob("*.jpg"):
                        if max_samples_per_class and nodog_count >= max_samples_per_class:
                            break
                        self.samples.append((str(img_file), 0))
                        nodog_count += 1
                        
            print(f"   ❌ NO-DOG: {nodog_count:,} images")
        
        # Load DOG images
        yesdog_path = self.data_path / "YESDOG"
        if yesdog_path.exists():
            dog_count = 0
            
            for breed_dir in yesdog_path.iterdir():
                if breed_dir.is_dir():
                    # Search for both .JPEG and .jpg extensions
                    for img_file in list(breed_dir.glob("*.JPEG")) + list(breed_dir.glob("*.jpg")):
                        if max_samples_per_class and dog_count >= max_samples_per_class:
                            break
                        self.samples.append((str(img_file), 1))
                        dog_count += 1
                        
            print(f"   ✅ DOG: {dog_count:,} images")
        
        # Balance dataset if limit specified
        if max_samples_per_class:
            self._balance_dataset(max_samples_per_class)
        
        print(f"🎯 Total samples: {len(self.samples):,}")
        
    def _balance_dataset(self, target_size):
        """Balance the dataset by limiting samples per class.
        
        Args:
            target_size: Target number of samples per class.
        """
        # Separate samples by class
        no_dog_samples = [s for s in self.samples if s[1] == 0]
        dog_samples = [s for s in self.samples if s[1] == 1]
        
        # Take target_size from each class
        no_dog_balanced = no_dog_samples[:target_size]
        dog_balanced = dog_samples[:target_size]
        
        self.samples = no_dog_balanced + dog_balanced
        
        print(f"⚖️  Dataset balanced: {len(no_dog_balanced)} no-dog + {len(dog_balanced)} dog")
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            tuple: (image_tensor, label) where label is 0 for no-dog, 1 for dog.
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            fallback = torch.zeros((3, 224, 224))
            return fallback, label

# ===================================================================
# BINARY MODEL
# ===================================================================


class BinaryDogClassifier(nn.Module):
    """EfficientNet-B1 based binary classifier for dog detection.
    
    Uses a pre-trained EfficientNet-B1 backbone with a custom classification head
    featuring dropout for regularization and batch normalization for stable training.
    
    Args:
        pretrained: Whether to use ImageNet pre-trained weights.
    
    Architecture:
        - EfficientNet-B1 backbone (frozen or fine-tunable)
        - Dropout (0.3) -> Linear (512) -> ReLU -> BatchNorm
        - Dropout (0.2) -> Linear (128) -> ReLU -> BatchNorm
        - Dropout (0.1) -> Linear (2) [output: no-dog, dog]
    """
    
    def __init__(self, pretrained=True):
        """Initialize the classifier with EfficientNet-B1 backbone."""
        super().__init__()
        
        # Load EfficientNet-B1 backbone with ImageNet pre-trained weights
        self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Get feature dimensions from the original classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom binary classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 2)  # Binary output: 0=no-dog, 1=dog
        )
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).
            
        Returns:
            Tensor: Logits of shape (batch_size, 2).
        """
        return self.backbone(x)

# ===================================================================
# TRAINER CLASS
# ===================================================================


class BinaryTrainer:
    """Training manager for binary dog classification.
    
    Handles the complete training pipeline including:
    - OneCycleLR learning rate scheduling
    - Gradient clipping for stable training
    - Early stopping with patience
    - Manual training control (interrupt safely)
    - Training history tracking and visualization
    
    Args:
        model: The neural network model to train.
        device: Computing device ('cpu' or 'cuda').
    
    Attributes:
        train_losses: History of training losses per epoch.
        train_accuracies: History of training accuracies per epoch.
        val_losses: History of validation losses per epoch.
        val_accuracies: History of validation accuracies per epoch.
    """
    
    def __init__(self, model, device='cpu'):
        """Initialize the trainer with model and device configuration."""
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.controller = TrainingController()  # Manual stop control
        
        # Configure AdamW optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Cross-entropy loss criterion
        self.criterion = nn.CrossEntropyLoss()
        
    def setup_scheduler(self, train_loader, epochs):
        """Configure the OneCycleLR learning rate scheduler.
        
        Args:
            train_loader: Training data loader (needed for total steps calculation).
            epochs: Total number of training epochs.
        """
        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.003,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        print(f"📈 OneCycleLR configured: {total_steps:,} total steps")
        
    def train_epoch(self, train_loader, epoch):
        """Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.
            
        Returns:
            tuple: (epoch_loss, epoch_accuracy, learning_rate)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Check for user stop request
            if self.controller.should_stop:
                print(f"\n🛑 Stop requested during training - finishing epoch...")
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar every 10 batches
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{current_lr:.2e}'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, current_lr
    
    def validate(self, val_loader, epoch):
        """Validate the model on the validation set.
        
        Args:
            val_loader: DataLoader for validation data.
            epoch: Current epoch number.
            
        Returns:
            tuple: (val_loss, val_accuracy, predictions, targets)
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Validation {epoch}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_preds, all_targets
    
    def train_model(self, train_loader, val_loader, epochs, save_path='./binary_models', patience=5):
        """Execute the complete training pipeline.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of training epochs.
            save_path: Directory to save model checkpoints.
            patience: Early stopping patience (epochs without improvement).
            
        Returns:
            dict: Training results including best accuracy, final epoch, and history.
        """
        print(f"🚀 STARTING BINARY TRAINING")
        print("=" * 60)
        print(f"🎯 Epochs: {epochs}")
        print(f"🤖 Model: EfficientNet-B1")
        print(f"💻 Device: {self.device}")
        print()
        
        # Start manual stop monitoring
        self.controller.start_monitoring()
        
        # Configure learning rate scheduler
        self.setup_scheduler(train_loader, epochs)
        
        # Create output directory
        Path(save_path).mkdir(exist_ok=True)
        
        best_val_acc = 0
        patience_counter = 0
        
        try:
            for epoch in range(1, epochs + 1):
                # Check for user stop request
                if self.controller.should_stop:
                    print(f"\n🛑 TRAINING STOPPED BY USER AT EPOCH {epoch}")
                    break
                
                print(f"📅 EPOCH {epoch}/{epochs}")
                print("-" * 40)
                
                # Train one epoch
                train_loss, train_acc, current_lr = self.train_epoch(train_loader, epoch)
                
                # Validate
                val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, epoch)
                
                # Record history
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Print epoch results
                print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"🔄 Learning Rate: {current_lr:.2e}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    # Save model checkpoint
                    model_path = Path(save_path) / 'best_binary_model.pth'
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'accuracy': val_acc,
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, model_path)
                
                    print(f"✅ Best model saved: {val_acc:.2f}% accuracy")
                else:
                    patience_counter += 1
                    print(f"⏳ Patience: {patience_counter}/{patience}")
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
                
                print()
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Training manually interrupted")
        finally:
            # Stop manual control monitoring
            self.controller.stop_monitoring()
        
        # Generate final training report
        self._generate_report(save_path, best_val_acc, val_preds, val_targets)
        
        return {
            'best_accuracy': best_val_acc,
            'final_epoch': epoch,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }
    
    def _generate_report(self, save_path, best_accuracy, preds, targets):
        """Generate training report with visualizations and metrics.
        
        Args:
            save_path: Directory to save the report files.
            best_accuracy: Best validation accuracy achieved.
            preds: Final validation predictions.
            targets: Ground truth labels.
        """
        print("📊 GENERATING FINAL REPORT...")
        
        # Create figure with training curves and confusion matrix
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(self.val_losses, label='Val Loss', color='red', alpha=0.7)
        plt.title('📉 Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train Acc', color='green', alpha=0.7)
        plt.plot(self.val_accuracies, label='Val Acc', color='orange', alpha=0.7)
        plt.title('📈 Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion matrix
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(targets, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No-Dog', 'Dog'],
                   yticklabels=['No-Dog', 'Dog'])
        plt.title('🎯 Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(Path(save_path) / 'binary_training_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Text classification report
        class_names = ['No-Dog', 'Dog']
        report = classification_report(targets, preds, target_names=class_names)
        
        with open(Path(save_path) / 'binary_classification_report.txt', 'w') as f:
            f.write("🐕 BINARY CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"🎯 Best Accuracy: {best_accuracy:.2f}%\n")
            f.write(f"💻 Optimized for: AMD 7800X3D\n")
            f.write(f"🤖 Architecture: EfficientNet-B1\n\n")
            f.write(report)
        
        print(f"✅ Report saved to {save_path}")

# ===================================================================
# DATA TRANSFORMS
# ===================================================================


def get_transforms():
    """Get image transformation pipelines for training and validation.
    
    Returns:
        tuple: (train_transform, val_transform) - Compose objects for each dataset.
        
    Training transforms include:
        - Resize to 256x256
        - Random crop to 224x224
        - Random horizontal flip
        - Random rotation (up to 15 degrees)
        - Color jitter augmentation
        - Normalization to ImageNet statistics
        
    Validation transforms include:
        - Resize to 256x256
        - Center crop to 224x224
        - Normalization to ImageNet statistics
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders(data_path, train_transform, val_transform, batch_size=16, num_workers=14):
    """Create optimized DataLoaders for training and validation.
    
    Args:
        data_path: Path to the dataset root directory.
        train_transform: Transforms for training data.
        val_transform: Transforms for validation data.
        batch_size: Batch size for both loaders.
        num_workers: Number of worker processes for data loading.
        
    Returns:
        tuple: (train_loader, val_loader) - DataLoader objects.
    """
    print("🔄 CREATING BINARY DATALOADERS...")
    
    # Create datasets with balanced sampling
    train_dataset = BinaryDogDataset(
        data_path=data_path,
        transform=train_transform,
        max_samples_per_class=15000  # 15k per class = 30k total
    )
    
    val_dataset = BinaryDogDataset(
        data_path=data_path,
        transform=val_transform,
        max_samples_per_class=3000   # 3k per class = 6k total
    )
    
    # Create DataLoaders optimized for AMD 7800X3D
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"✅ DataLoaders created:")
    print(f"   🏋️  Train: {len(train_dataset):,} samples")
    print(f"   ✅ Val: {len(val_dataset):,} samples")
    print(f"   ⚙️  Batch size: {batch_size}")
    print(f"   👷 Workers: {num_workers}")
    
    return train_loader, val_loader

# ===================================================================
# MAIN ENTRY POINT
# ===================================================================


def main():
    """Main entry point for binary dog classifier training.
    
    Executes the complete training pipeline:
    1. Configure environment for AMD 7800X3D
    2. Load and prepare dataset
    3. Create model and trainer
    4. Train with early stopping
    5. Generate and save reports
    
    Returns:
        dict: Training results or None if dataset not found.
    """
    print("🐕 BINARY DOG TRAINER")
    print("🚀 Optimized for AMD 7800X3D")
    print("=" * 80)
    
    # Configure environment for optimal performance
    optimize_for_7800x3d()
    
    # Training configuration
    DATA_PATH = "./DATASETS"
    BATCH_SIZE = 16  # Optimized for memory efficiency
    EPOCHS = 20
    NUM_WORKERS = 14  # 7800X3D has 16 threads, leave 2 for system
    
    # Verify dataset exists
    if not Path(DATA_PATH).exists():
        print(f"❌ Data directory not found: {DATA_PATH}")
        return
    
    # Create model
    print("🤖 Creating model...")
    model = BinaryDogClassifier(pretrained=True)
    
    # Configure computing device
    device = torch.device('cpu')
    print(f"💻 Using device: {device}")
    
    # Create trainer instance
    trainer = BinaryTrainer(model, device)
    
    # Prepare data loaders
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = create_dataloaders(
        DATA_PATH, train_transform, val_transform, BATCH_SIZE, NUM_WORKERS
    )
    
    # Execute training
    start_time = time.time()
    results = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_path='./binary_models',
        patience=5
    )
    
    training_time = time.time() - start_time
    
    # Print final results
    print("\n🎯 TRAINING COMPLETED")
    print("=" * 60)
    print(f"✅ Best accuracy: {results['best_accuracy']:.2f}%")
    print(f"⏱️  Total time: {training_time/3600:.1f} hours")
    print(f"📊 Epochs completed: {results['final_epoch']}")
    print(f"💾 Model saved: ./binary_models/best_binary_model.pth")
    
    return results


if __name__ == "__main__":
    results = main()