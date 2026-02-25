"""
Data Preprocessor for Binary Dog Classification.

This module provides comprehensive data preprocessing capabilities for binary
dog vs non-dog image classification. Optimized for GPU training on AMD 7900XTX.

Key Components:
    - DogClassificationDataset: Custom PyTorch Dataset for binary classification
    - DataPreprocessor: Main preprocessing pipeline with augmentation support
    - create_sample_visualization: Visualization utility for augmented samples

Features:
    - Automatic image collection from structured directories
    - Class balancing via undersampling or oversampling
    - Advanced data augmentation using Albumentations
    - Stratified train/validation/test splitting
    - Optimized DataLoader creation for GPU training

Usage:
    preprocessor = DataPreprocessor(dataset_path, output_path)
    data_loaders, splits = preprocessor.process_complete_dataset()

Author: Dog Classification Project Team
Hardware Optimization: AMD 7800X3D CPU / AMD 7900XTX GPU
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import json
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class DogClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for binary dog classification.
    
    Handles image loading, transformation, and label association for
    training binary classifiers to distinguish dogs from non-dogs.
    
    Attributes:
        image_paths (list): List of Path objects to image files.
        labels (list): Binary labels (1=dog, 0=non-dog).
        transform: Albumentations transformation pipeline.
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset with image paths and labels.
        
        Args:
            image_paths (list): List of Path objects pointing to images.
            labels (list): Binary labels corresponding to each image.
            transform: Optional Albumentations transform pipeline.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (image_tensor, label_tensor) where image is transformed
                   and label is a float tensor for BCE loss compatibility.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image using OpenCV and convert to RGB
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to black image on error
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, torch.tensor(label, dtype=torch.float32)


class DataPreprocessor:
    """
    Main preprocessing pipeline for dog classification datasets.
    
    Provides end-to-end data preparation including collection, balancing,
    splitting, augmentation, and DataLoader creation.
    
    Attributes:
        dataset_path (Path): Root path containing YESDOG and NODOG folders.
        output_path (Path): Directory for saving processed data and reports.
        target_size (tuple): Target image dimensions (height, width).
        yesdog_path (Path): Path to dog images directory.
        nodog_path (Path): Path to non-dog images directory.
        image_extensions (set): Supported image file extensions.
        imagenet_mean (list): ImageNet normalization mean values.
        imagenet_std (list): ImageNet normalization std values.
    """
    
    def __init__(self, dataset_path: str, output_path: str, target_size: tuple = (224, 224)):
        """
        Initialize the preprocessor with dataset paths.
        
        Args:
            dataset_path (str): Path to root dataset directory.
            output_path (str): Path for saving processed data.
            target_size (tuple): Target image size, default (224, 224).
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.yesdog_path = self.dataset_path / "YESDOG"
        self.nodog_path = self.dataset_path / "NODOG"
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image file extensions for filtering
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # ImageNet normalization statistics for transfer learning
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
    def collect_all_images(self):
        """
        Collect all image paths and their corresponding labels.
        
        Traverses the YESDOG and NODOG directories to build complete
        lists of image paths with binary labels.
        
        Returns:
            tuple: (image_paths, labels) where image_paths is a list of
                   Path objects and labels is a list of binary integers
                   (1=dog, 0=non-dog).
        """
        print("📂 Collecting image paths...")
        
        image_paths = []
        labels = []
        
        # Dog images (label 1)
        print("   Processing dog images...")
        dog_count = 0
        for breed_folder in tqdm(list(self.yesdog_path.iterdir())):
            if breed_folder.is_dir():
                for img_file in breed_folder.iterdir():
                    if img_file.suffix.lower() in self.image_extensions:
                        if self._is_valid_image(img_file):
                            image_paths.append(img_file)
                            labels.append(1)  # Dog
                            dog_count += 1
        
        # Non-dog images (label 0)
        print("   Processing non-dog images...")
        nodog_count = 0
        for category_folder in tqdm(list(self.nodog_path.iterdir())):
            if category_folder.is_dir():
                for img_file in category_folder.iterdir():
                    if img_file.suffix.lower() in self.image_extensions:
                        if self._is_valid_image(img_file):
                            image_paths.append(img_file)
                            labels.append(0)  # No-dog
                            nodog_count += 1
        
        print(f"✅ Collection completed:")
        print(f"   - Dog images: {dog_count:,}")
        print(f"   - Non-dog images: {nodog_count:,}")
        print(f"   - Total: {len(image_paths):,}")
        print(f"   - Dog/non-dog ratio: {dog_count/max(nodog_count, 1):.2f}")
        
        return image_paths, labels
    
    def _is_valid_image(self, image_path: Path) -> bool:
        """
        Validate that an image file can be loaded correctly.
        
        Attempts to read the image to verify it's not corrupted.
        
        Args:
            image_path (Path): Path to the image file to validate.
            
        Returns:
            bool: True if image loads successfully with valid dimensions,
                  False otherwise.
        """
        try:
            img = cv2.imread(str(image_path))
            return img is not None and img.shape[0] > 0 and img.shape[1] > 0
        except:
            return False
    
    def balance_classes(self, image_paths: list, labels: list, strategy: str = 'undersample'):
        """
        Balance the class distribution in the dataset.
        
        Applies undersampling or oversampling to achieve equal class sizes.
        
        Args:
            image_paths (list): List of image path objects.
            labels (list): List of binary labels.
            strategy (str): Balancing strategy - 'undersample' or 'oversample'.
            
        Returns:
            tuple: (balanced_paths, balanced_labels) with equal class counts.
        """
        print(f"⚖️ Balancing classes with strategy: {strategy}")
        
        # Separate indices by class
        dog_indices = [i for i, label in enumerate(labels) if label == 1]
        nodog_indices = [i for i, label in enumerate(labels) if label == 0]
        
        dog_count = len(dog_indices)
        nodog_count = len(nodog_indices)
        
        print(f"   Before - Dogs: {dog_count:,}, Non-dogs: {nodog_count:,}")
        
        if strategy == 'undersample':
            # Reduce the majority class to match minority
            target_size = min(dog_count, nodog_count)
            
            if dog_count > target_size:
                dog_indices = random.sample(dog_indices, target_size)
            if nodog_count > target_size:
                nodog_indices = random.sample(nodog_indices, target_size)
                
        elif strategy == 'oversample':
            # Increase the minority class by duplicating images
            target_size = max(dog_count, nodog_count)
            
            if dog_count < target_size:
                needed = target_size - dog_count
                dog_indices.extend(random.choices(dog_indices, k=needed))
            if nodog_count < target_size:
                needed = target_size - nodog_count
                nodog_indices.extend(random.choices(nodog_indices, k=needed))
        
        # Reconstruct balanced lists
        balanced_indices = dog_indices + nodog_indices
        balanced_paths = [image_paths[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        # Shuffle combined data
        combined = list(zip(balanced_paths, balanced_labels))
        random.shuffle(combined)
        balanced_paths, balanced_labels = zip(*combined)
        
        print(f"   After - Dogs: {balanced_labels.count(1):,}, Non-dogs: {balanced_labels.count(0):,}")
        
        return list(balanced_paths), list(balanced_labels)
    
    def create_train_val_test_split(self, image_paths: list, labels: list, 
                                  train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Split the dataset into training, validation, and test sets.
        
        Uses stratified splitting to maintain class proportions across splits.
        
        Args:
            image_paths (list): List of image path objects.
            labels (list): List of binary labels.
            train_ratio (float): Proportion for training set, default 0.7.
            val_ratio (float): Proportion for validation set, default 0.15.
            
        Returns:
            dict: Dictionary with 'train', 'val', 'test' keys, each containing
                  'paths' and 'labels' lists.
        """
        print(f"📊 Splitting dataset: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%}")
        
        # First split: separate training from temp (validation + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=(1-train_ratio), 
            random_state=42, stratify=labels
        )
        
        # Second split: separate temp into validation and test
        val_size = val_ratio / (val_ratio + (1-train_ratio-val_ratio))
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=(1-val_size),
            random_state=42, stratify=temp_labels
        )
        
        splits = {
            'train': {'paths': train_paths, 'labels': train_labels},
            'val': {'paths': val_paths, 'labels': val_labels},
            'test': {'paths': test_paths, 'labels': test_labels}
        }
        
        for split_name, split_data in splits.items():
            dog_count = split_data['labels'].count(1)
            nodog_count = split_data['labels'].count(0)
            total = len(split_data['labels'])
            print(f"   {split_name.upper():5s}: {total:5,} images (dogs: {dog_count:,}, non-dogs: {nodog_count:,})")
        
        return splits
    
    def get_augmentation_transforms(self, mode: str = 'train'):
        """
        Get Albumentations transformation pipeline for the specified mode.
        
        Creates different augmentation strategies for training vs evaluation.
        Training includes aggressive augmentation while validation/test only
        applies normalization.
        
        Args:
            mode (str): Either 'train' for training augmentations or any other
                        value for minimal validation transforms.
                        
        Returns:
            A.Compose: Albumentations composition of transforms.
        """
        
        if mode == 'train':
            # Aggressive augmentation for training robustness
            transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.RandomCrop(height=int(self.target_size[0]*0.9), 
                           width=int(self.target_size[1]*0.9), p=0.3),
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.2),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
                ToTensorV2()
            ])
        else:
            # Minimal transforms for validation and test
            transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
                ToTensorV2()
            ])
        
        return transform
    
    def create_data_loaders(self, splits: dict, batch_size: int = 32, num_workers: int = 4):
        """
        Create PyTorch DataLoaders for all dataset splits.
        
        Configures optimized DataLoaders with pin_memory for GPU training.
        
        Args:
            splits (dict): Dictionary from create_train_val_test_split.
            batch_size (int): Batch size for all loaders, default 32.
            num_workers (int): Number of data loading workers, default 4.
            
        Returns:
            dict: Dictionary with 'train', 'val', 'test' DataLoader objects.
        """
        print(f"🔄 Creating DataLoaders (batch_size={batch_size}, num_workers={num_workers})...")
        
        # Get transforms for each mode
        train_transform = self.get_augmentation_transforms('train')
        val_transform = self.get_augmentation_transforms('val')
        
        # Create Dataset objects
        train_dataset = DogClassificationDataset(
            splits['train']['paths'], 
            splits['train']['labels'], 
            transform=train_transform
        )
        
        val_dataset = DogClassificationDataset(
            splits['val']['paths'], 
            splits['val']['labels'], 
            transform=val_transform
        )
        
        test_dataset = DogClassificationDataset(
            splits['test']['paths'], 
            splits['test']['labels'], 
            transform=val_transform
        )
        
        # Create DataLoaders with GPU optimization
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✅ DataLoaders created:")
        print(f"   - Train: {len(train_loader)} batches")
        print(f"   - Val:   {len(val_loader)} batches")
        print(f"   - Test:  {len(test_loader)} batches")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def save_preprocessing_info(self, splits: dict):
        """
        Save preprocessing configuration and statistics to JSON file.
        
        Exports dataset statistics and preprocessing configuration for
        reproducibility and documentation.
        
        Args:
            splits (dict): Dataset splits dictionary with paths and labels.
            
        Returns:
            None: Saves 'preprocessing_info.json' to output directory.
        """
        info = {
            'dataset_stats': {
                'total_images': sum(len(split['labels']) for split in splits.values()),
                'train_images': len(splits['train']['labels']),
                'val_images': len(splits['val']['labels']),
                'test_images': len(splits['test']['labels']),
            },
            'class_distribution': {
                'train': {
                    'dogs': splits['train']['labels'].count(1),
                    'no_dogs': splits['train']['labels'].count(0)
                },
                'val': {
                    'dogs': splits['val']['labels'].count(1),
                    'no_dogs': splits['val']['labels'].count(0)
                },
                'test': {
                    'dogs': splits['test']['labels'].count(1),
                    'no_dogs': splits['test']['labels'].count(0)
                }
            },
            'preprocessing_config': {
                'target_size': self.target_size,
                'normalization_mean': self.imagenet_mean,
                'normalization_std': self.imagenet_std,
                'augmentation_enabled': True
            }
        }
        
        info_path = self.output_path / 'preprocessing_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"💾 Information saved to: {info_path}")
        
    def process_complete_dataset(self, balance_strategy: str = 'undersample', 
                               batch_size: int = 32):
        """
        Execute the complete dataset preprocessing pipeline.
        
        Orchestrates all preprocessing steps in sequence: collection,
        balancing, splitting, DataLoader creation, and report saving.
        
        Args:
            balance_strategy (str): 'undersample', 'oversample', or None.
            batch_size (int): Batch size for DataLoaders, default 32.
            
        Returns:
            tuple: (data_loaders, splits) - DataLoader dict and splits dict.
        """
        print("🚀 Starting complete preprocessing...")
        print("="*60)
        
        # 1. Collect all images
        image_paths, labels = self.collect_all_images()
        
        # 2. Balance classes if strategy specified
        if balance_strategy:
            image_paths, labels = self.balance_classes(image_paths, labels, balance_strategy)
        
        # 3. Split into train/val/test
        splits = self.create_train_val_test_split(image_paths, labels)
        
        # 4. Create DataLoaders
        data_loaders = self.create_data_loaders(splits, batch_size=batch_size)
        
        # 5. Save preprocessing information
        self.save_preprocessing_info(splits)
        
        print("\n🎉 Preprocessing completed successfully!")
        return data_loaders, splits


def create_sample_visualization(data_loaders, save_path: str):
    """
    Create a visualization grid of augmented training samples.
    
    Generates a 4x4 grid showing samples from the training set with
    their labels, useful for verifying augmentation effects.
    
    Args:
        data_loaders (dict): Dictionary containing 'train' DataLoader.
        save_path (str): File path to save the visualization image.
        
    Returns:
        None: Saves visualization to specified path and displays it.
    """
    import matplotlib.pyplot as plt
    
    print("📸 Creating sample visualization...")
    
    # Get a batch from training loader
    train_loader = data_loaders['train']
    batch_iter = iter(train_loader)
    images, labels = next(batch_iter)
    
    # ImageNet denormalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Dataset Samples with Augmentation', fontsize=16)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # Denormalize image for visualization
        img = images[i].clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = torch.clamp(img, 0, 1)
        
        # Convert tensor to numpy array
        img_np = img.permute(1, 2, 0).numpy()
        
        # Display image
        axes[row, col].imshow(img_np)
        label_text = "🐕 DOG" if labels[i].item() == 1 else "📦 NON-DOG"
        axes[row, col].set_title(label_text, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved to: {save_path}")


# Main execution block when script is run directly
if __name__ == "__main__":
    # Configuration paths
    dataset_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS"
    output_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\processed_data"
    
    # Create preprocessor instance
    preprocessor = DataPreprocessor(dataset_path, output_path)
    
    # Process the complete dataset
    data_loaders, splits = preprocessor.process_complete_dataset(
        balance_strategy='undersample',  # Options: 'undersample', 'oversample', or None
        batch_size=32
    )
    
    # Create visualization of augmented samples
    sample_viz_path = Path(output_path) / 'sample_visualization.png'
    create_sample_visualization(data_loaders, str(sample_viz_path))