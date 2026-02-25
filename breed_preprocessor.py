"""
Top 50 Dog Breeds Dataset Preprocessor
======================================

Specialized preprocessor for creating balanced training datasets from the
top 50 dog breeds. Optimized for AMD Ryzen 7 7800X3D with intelligent
balancing strategies.

Features:
---------
- Analyze breed distribution and class imbalance
- Create balanced datasets via undersampling/oversampling
- Advanced data augmentation with Albumentations
- Train/val/test splits with stratification
- Optimized DataLoaders for 7800X3D parallelism
- Weighted random sampling for balanced training

Classes:
--------
- BreedDatasetPreprocessor: Main preprocessing pipeline
- BreedDataset: PyTorch Dataset for breed images

Author: Dog Breed Classifier Team
Date: 2024
"""

import os
import json
import time
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BreedDatasetPreprocessor:
    """
    Preprocessor for creating balanced dog breed datasets.
    
    Handles dataset analysis, balancing strategies, data augmentation,
    and DataLoader creation optimized for AMD 7800X3D processor.
    
    Attributes:
        yesdog_path (Path): Path to dog breed images.
        output_path (Path): Path for processed output.
        breed_config (dict): Configuration for top 50 breeds.
        name_to_index (dict): Breed name to class index mapping.
        index_to_display (dict): Class index to display name mapping.
        cpu_config (dict): DataLoader configuration for 7800X3D.
    """
    
    def __init__(self, yesdog_path: str, output_path: str = "./breed_processed_data"):
        """
        Initialize the breed dataset preprocessor.
        
        Args:
            yesdog_path (str): Path to YESDOG directory with breed folders.
            output_path (str): Output directory for processed data.
        """
        self.yesdog_path = Path(yesdog_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Load breed configuration
        try:
            from breed_config import TOP_50_BREEDS, BREED_NAME_TO_INDEX, BREED_INDEX_TO_DISPLAY
            self.breed_config = TOP_50_BREEDS
            self.name_to_index = BREED_NAME_TO_INDEX
            self.index_to_display = BREED_INDEX_TO_DISPLAY
            print("✅ Breed configuration loaded")
        except ImportError:
            print("❌ Error: Run top_50_selector.py first")
            raise
        
        # CPU configuration optimized for 7800X3D
        self.cpu_config = {
            'batch_size': 16,
            'num_workers': 14,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
        }
        
        # Set environment variables
        self.setup_environment()
        
    def setup_environment(self):
        """
        Configure environment variables for AMD 7800X3D optimization.
        
        Sets thread counts for parallel libraries (OpenMP, MKL, NumExpr)
        and enables PyTorch JIT optimizations.
        """
        env_vars = {
            'OMP_NUM_THREADS': '16',
            'MKL_NUM_THREADS': '16',
            'NUMEXPR_NUM_THREADS': '16',
            'OPENBLAS_NUM_THREADS': '16',
            'VECLIB_MAXIMUM_THREADS': '16',
            'PYTORCH_JIT': '1',
            'PYTORCH_JIT_OPT_LEVEL': '2'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        print("🚀 7800X3D environment variables configured")
        
    def analyze_breed_distribution(self):
        """
        Analyze the distribution of images across dog breeds.
        
        Scans breed directories to count actual images and compares
        against expected counts from configuration.
        
        Returns:
            tuple: (breed_stats dict, total_images count)
        """
        print("\n🔍 ANALYZING BREED DISTRIBUTION...")
        print("="*60)
        
        breed_stats = {}
        total_images = 0
        
        for breed_info in self.breed_config['breeds'].values():
            breed_name = breed_info['name']
            original_dir = breed_info['original_dir']
            expected_count = breed_info['image_count']
            
            breed_path = self.yesdog_path / original_dir
            if breed_path.exists():
                # Count actual files
                image_files = [f for f in breed_path.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                actual_count = len(image_files)
                
                breed_stats[breed_name] = {
                    'path': breed_path,
                    'expected': expected_count,
                    'actual': actual_count,
                    'class_index': breed_info['class_index'],
                    'display_name': breed_info['display_name'],
                    'files': image_files
                }
                total_images += actual_count
            else:
                print(f"⚠️  Breed not found: {breed_name} ({original_dir})")
        
        # Calculate distribution statistics
        counts = [info['actual'] for info in breed_stats.values()]
        min_count = min(counts)
        max_count = max(counts)
        mean_count = np.mean(counts)
        
        print(f"📊 Breeds processed: {len(breed_stats)}")
        print(f"📊 Total images: {total_images:,}")
        print(f"📈 Range: {min_count} - {max_count} images")
        print(f"📊 Average: {mean_count:.1f} images per breed")
        print(f"⚖️  Imbalance ratio: {max_count/min_count:.2f}x")
        
        return breed_stats, total_images
    
    def create_balanced_strategy(self, breed_stats: dict, target_samples_per_class: int = 200):
        """
        Create intelligent balancing strategy for dataset.
        
        Determines whether each breed needs undersampling (too many images)
        or oversampling with augmentation (too few images).
        
        Args:
            breed_stats (dict): Statistics for each breed.
            target_samples_per_class (int): Target number of samples per class.
            
        Returns:
            dict: Balancing strategy for each breed.
        """
        print(f"\n⚖️  CREATING BALANCING STRATEGY...")
        print(f"   🎯 Target: {target_samples_per_class} samples per breed")
        print("="*60)
        
        balance_strategy = {}
        
        for breed_name, info in breed_stats.items():
            actual_count = info['actual']
            class_index = info['class_index']
            
            if actual_count >= target_samples_per_class:
                # Undersample: select randomly
                strategy = {
                    'type': 'undersample',
                    'original_count': actual_count,
                    'target_count': target_samples_per_class,
                    'factor': target_samples_per_class / actual_count
                }
            else:
                # Oversample: augmentation needed
                augmentation_factor = max(1, target_samples_per_class // actual_count)
                remaining = target_samples_per_class - (actual_count * augmentation_factor)
                
                strategy = {
                    'type': 'oversample',
                    'original_count': actual_count,
                    'target_count': target_samples_per_class,
                    'augmentation_factor': augmentation_factor,
                    'remaining_samples': max(0, remaining),
                    'total_factor': target_samples_per_class / actual_count
                }
            
            balance_strategy[breed_name] = strategy
        
        # Print strategy summary
        undersample_count = sum(1 for s in balance_strategy.values() if s['type'] == 'undersample')
        oversample_count = sum(1 for s in balance_strategy.values() if s['type'] == 'oversample')
        
        print(f"📉 Breeds to undersample: {undersample_count}")
        print(f"📈 Breeds to oversample: {oversample_count}")
        print(f"🎯 Total samples after balancing: {len(balance_strategy) * target_samples_per_class:,}")
        
        return balance_strategy
    
    def create_advanced_augmentations(self):
        """
        Create advanced data augmentation pipelines.
        
        Returns separate transform pipelines for train, validation, and test:
        - Train: Aggressive augmentations (flips, rotations, color, noise)
        - Val/Test: Only resize, crop, and normalization
        
        Returns:
            dict: Transform pipelines keyed by 'train', 'val', 'test'.
        """
        
        # Training augmentations (aggressive)
        train_transform = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(height=224, width=224),
            
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.7
            ),
            
            # Color augmentations
            A.OneOf([
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.8),
            
            # Noise and blur augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Occlusions and dropout
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=128,
                p=0.3
            ),
            
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.2),
            
            # ImageNet normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # Validation transforms (only resize and normalize)
        val_transform = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # Test transforms (same as validation)
        test_transform = val_transform
        
        return {
            'train': train_transform,
            'val': val_transform,
            'test': test_transform
        }
    
    def create_balanced_dataset(self, breed_stats: dict, balance_strategy: dict, 
                               test_size: float = 0.2, val_size: float = 0.15):
        """
        Create balanced dataset with train/val/test splits.
        
        Applies balancing strategy to each breed and creates proper splits
        ensuring augmented images only appear in training set.
        
        Args:
            breed_stats (dict): Statistics for each breed.
            balance_strategy (dict): Balancing strategy per breed.
            test_size (float): Fraction for test split.
            val_size (float): Fraction for validation split.
            
        Returns:
            dict: Dataset information including counts per split.
        """
        print(f"\n📂 CREATING BALANCED DATASET...")
        print("="*60)
        
        # Create directories of output
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = self.output_path / split
            split_dir.mkdir(exist_ok=True)
            
            for breed_name in breed_stats.keys():
                breed_dir = split_dir / breed_name
                breed_dir.mkdir(exist_ok=True)
        
        # Process each breed
        dataset_info = {}
        total_processed = 0
        
        for breed_name, info in breed_stats.items():
            print(f"   📝 Processing: {info['display_name']}")
            
            strategy = balance_strategy[breed_name]
            files = info['files']
            
            # Apply balancing strategy
            if strategy['type'] == 'undersample':
                # Select randomly
                target_count = strategy['target_count']
                selected_files = random.sample(files, min(target_count, len(files)))
            else:  # oversample
                # Use all files + augmentations
                selected_files = files.copy()
                
                # Calculate additional augmentations needed
                target_count = strategy['target_count']
                original_count = len(selected_files)
                needed_augmentations = max(0, target_count - original_count)
                
                # Create additional augmentations if needed
                if needed_augmentations > 0:
                    # Create list of files to augment cyclically
                    files_to_augment = []
                    for i in range(needed_augmentations):
                        file_idx = i % len(selected_files)
                        files_to_augment.append(selected_files[file_idx])
                    
                    # Mark augmented files
                    augmented_files = [(f, True) for f in files_to_augment]
                    original_files = [(f, False) for f in selected_files]
                    all_files = original_files + augmented_files
                else:
                    all_files = [(f, False) for f in selected_files]
                
                selected_files = all_files
            
            # Split train/val/test
            if strategy['type'] == 'undersample':
                # Split normal for undersampled
                train_files, temp_files = train_test_split(
                    selected_files, test_size=(test_size + val_size), random_state=42
                )
                val_files, test_files = train_test_split(
                    temp_files, test_size=(test_size / (test_size + val_size)), random_state=42
                )
            else:
                # For oversampled: separate original from augmented
                original_files = [f for f, is_aug in selected_files if not is_aug]
                augmented_files = [f for f, is_aug in selected_files if is_aug]
                
                # Split original files
                train_orig, temp_orig = train_test_split(
                    original_files, test_size=(test_size + val_size), random_state=42
                )
                val_orig, test_orig = train_test_split(
                    temp_orig, test_size=(test_size / (test_size + val_size)), random_state=42
                )
                
                # All augmented images go to train
                train_files = train_orig + [(f, True) for f in augmented_files]
                val_files = val_orig
                test_files = test_orig
            
            # Store dataset info for this breed
            dataset_info[breed_name] = {
                'class_index': info['class_index'],
                'display_name': info['display_name'],
                'train_count': len(train_files),
                'val_count': len(val_files),
                'test_count': len(test_files),
                'total_count': len(train_files) + len(val_files) + len(test_files),
                'strategy': strategy
            }
            
            # Copy files to their corresponding directories
            self.copy_files_to_splits(breed_name, train_files, val_files, test_files)
            
            total_processed += dataset_info[breed_name]['total_count']
            
        print(f"\n✅ Balanced dataset created:")
        print(f"   📊 Total processed: {total_processed:,} images")
        print(f"   🏷️  Breeds: {len(dataset_info)}")
        
        # Save dataset configuration
        self.save_dataset_info(dataset_info)
        
        return dataset_info
    
    def copy_files_to_splits(self, breed_name: str, train_files, val_files, test_files):
        """
        Copy files to their corresponding split directories.
        
        Handles both original and augmented files, applying unique naming
        for augmented copies.
        
        Args:
            breed_name (str): Name of the breed.
            train_files: List of training files (may include tuples for augmented).
            val_files: List of validation files.
            test_files: List of test files.
            
        Returns:
            tuple: (train_copied, val_copied, test_copied) counts.
        """
        
        def copy_file_list(file_list, split_name):
            split_dir = self.output_path / split_name / breed_name
            copied = 0
            
            for item in file_list:
                if isinstance(item, tuple):
                    file_path, is_augmented = item
                    if is_augmented:
                        # Rename augmented files with unique suffix
                        base_name = file_path.stem
                        extension = file_path.suffix
                        new_name = f"{base_name}_aug_{copied}{extension}"
                    else:
                        new_name = file_path.name
                else:
                    file_path = item
                    new_name = file_path.name
                
                dst_path = split_dir / new_name
                try:
                    shutil.copy2(file_path, dst_path)
                    copied += 1
                except Exception as e:
                    print(f"   ⚠️  Error copying {file_path}: {e}")
            
            return copied
        
        # Copy files to each split
        train_copied = copy_file_list(train_files, 'train')
        val_copied = copy_file_list(val_files, 'val')
        test_copied = copy_file_list(test_files, 'test')
        
        return train_copied, val_copied, test_copied
    
    def save_dataset_info(self, dataset_info: dict):
        """
        Save dataset information and configuration files.
        
        Creates JSON summary and Python configuration module for
        easy import during training.
        
        Args:
            dataset_info (dict): Dataset statistics and configuration.
        """
        
        # Create summary
        summary = {
            'total_breeds': len(dataset_info),
            'total_train': sum(info['train_count'] for info in dataset_info.values()),
            'total_val': sum(info['val_count'] for info in dataset_info.values()),
            'total_test': sum(info['test_count'] for info in dataset_info.values()),
            'breed_details': dataset_info
        }
        
        summary['total_images'] = summary['total_train'] + summary['total_val'] + summary['total_test']
        
        # Save as JSON
        with open(self.output_path / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate Python configuration module
        config_py = f"""# Auto-generated dataset configuration
# Generated by BreedDatasetPreprocessor

DATASET_INFO = {dataset_info}

DATASET_SUMMARY = {summary}

# Quick lookup mappings
BREED_TO_INDEX = {{
"""
        
for breed_name, info in dataset_info.items():
config_py += f' "{breed_name}": {info["class_index"]},\n'
        
config_py += "}\n\nINDEX_TO_BREED = {\n"
        
for breed_name, info in dataset_info.items():
config_py += f' {info["class_index"]}: "{breed_name}",\n'
        
config_py += "}\n\nINDEX_TO_DISPLAY = {\n"
        
for breed_name, info in dataset_info.items():
config_py += f' {info["class_index"]}: "{info["display_name"]}",\n'
        
config_py += "}\n"
        
with open(self.output_path / 'dataset_config.py', 'w', encoding='utf-8') as f:
f.write(config_py)
        
print(f" 💾 Saved: dataset_info.json")
print(f" 💾 Saved: dataset_config.py")
        
def create_data_loaders(self, dataset_info: dict):
        """
        Create optimized DataLoaders for training.
        
        Creates DataLoaders with weighted random sampling for balanced
        training and proper configuration for 7800X3D parallelism.
        
        Args:
            dataset_info (dict): Dataset statistics and configuration.
            
        Returns:
            tuple: (data_loaders dict, datasets dict)
        """
        print(f"\n🔄 CREATING OPTIMIZED DATALOADERS...")
        print("="*60)
        
        # Get transformaciones
        transforms_dict = self.create_advanced_augmentations()
        
        # Create datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            split_dir = self.output_path / split
            datasets[split] = BreedDataset(
                root_dir=split_dir,
                transform=transforms_dict[split],
                breed_to_index=self.name_to_index
            )
        
        # Create weighted sampler for training
        train_targets = [datasets['train'][i][1] for i in range(len(datasets['train']))]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_targets),
            y=train_targets
        )
        
        sample_weights = [class_weights[t] for t in train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create DataLoaders
        data_loaders = {}
        
        # Training loader with sampler
        data_loaders['train'] = DataLoader(
            datasets['train'],
            batch_size=self.cpu_config['batch_size'],
            sampler=sampler,
            num_workers=self.cpu_config['num_workers'],
            pin_memory=self.cpu_config['pin_memory'],
            persistent_workers=self.cpu_config['persistent_workers'],
            prefetch_factor=self.cpu_config['prefetch_factor']
        )
        
        # Validation and Test loaders
        for split in ['val', 'test']:
            data_loaders[split] = DataLoader(
                datasets[split],
                batch_size=self.cpu_config['batch_size'],
                shuffle=False,
                num_workers=self.cpu_config['num_workers'],
                pin_memory=self.cpu_config['pin_memory'],
                persistent_workers=self.cpu_config['persistent_workers'],
                prefetch_factor=self.cpu_config['prefetch_factor']
            )
        
        print(f"✅ DataLoaders created:")
        print(f"   🏋️  Train: {len(datasets['train']):,} samples")
        print(f"   ✅ Val: {len(datasets['val']):,} samples")
        print(f"   🧪 Test: {len(datasets['test']):,} samples")
        print(f"   ⚙️  Batch size: {self.cpu_config['batch_size']}")
        print(f"   👷 Workers: {self.cpu_config['num_workers']}")
        
        return data_loaders, datasets
    
    def run_complete_preprocessing(self, target_samples_per_class: int = 200):
        """
        Execute complete preprocessing pipeline.
        
        Runs all preprocessing steps: analysis, balancing, dataset creation,
        and DataLoader setup.
        
        Args:
            target_samples_per_class (int): Target samples per class after balancing.
            
        Returns:
            dict: Complete results including loaders, datasets, and timing.
        """
        start_time = time.time()
        
        print("🎯 COMPLETE BREED PREPROCESSING")
        print("="*80)
        print(f"🎯 Target: {target_samples_per_class} samples per class")
        print(f"💻 Optimized for: AMD 7800X3D")
        
        # 1. Analyze breed distribution
        breed_stats, total_images = self.analyze_breed_distribution()
        
        # 2. Create balancing strategy
        balance_strategy = self.create_balanced_strategy(breed_stats, target_samples_per_class)
        
        # 3. Create balanced dataset
        dataset_info = self.create_balanced_dataset(breed_stats, balance_strategy)
        
        # 4. Create DataLoaders
        data_loaders, datasets = self.create_data_loaders(dataset_info)
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print(f"\n🎯 FINAL SUMMARY:")
        print("="*60)
        print(f"✅ Preprocessing completed in {elapsed_time:.1f} seconds")
        print(f"🏷️  Breeds processed: {len(dataset_info)}")
        print(f"📊 Total images: {sum(info['total_count'] for info in dataset_info.values()):,}")
        print(f"🏋️  Training: {sum(info['train_count'] for info in dataset_info.values()):,}")
        print(f"✅ Validation: {sum(info['val_count'] for info in dataset_info.values()):,}")
        print(f"🧪 Test: {sum(info['test_count'] for info in dataset_info.values()):,}")
        print(f"💻 Optimized for 7800X3D: ✅"))
        
        return {
            'data_loaders': data_loaders,
            'datasets': datasets,
            'dataset_info': dataset_info,
            'breed_stats': breed_stats,
            'preprocessing_time': elapsed_time
        }

class BreedDataset(Dataset):
    """
    Custom PyTorch Dataset for dog breed images.
    
    Loads images from directory structure where each subdirectory
    represents a breed class.
    
    Attributes:
        root_dir (Path): Root directory containing breed folders.
        transform: Transform pipeline to apply to images.
        breed_to_index (dict): Breed name to class index mapping.
        samples (list): List of (image_path, label) tuples.
        classes (list): List of (breed_name, class_index) tuples.
    """
    
    def __init__(self, root_dir, transform=None, breed_to_index=None):
        """
        Initialize the breed dataset.
        
        Args:
            root_dir: Root directory with breed subdirectories.
            transform: Transform pipeline for images.
            breed_to_index (dict): Optional breed name to index mapping.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.breed_to_index = breed_to_index or {}
        
        # Find all images
        self.samples = []
        self.classes = []
        
        for breed_dir in self.root_dir.iterdir():
            if breed_dir.is_dir():
                breed_name = breed_dir.name
                class_index = self.breed_to_index.get(breed_name, len(self.classes))
                
                if breed_name not in [c[0] for c in self.classes]:
                    self.classes.append((breed_name, class_index))
                
                # Find images
                for img_path in breed_dir.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append((img_path, class_index))
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index.
            
        Returns:
            tuple: (image tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback image
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transformaciones
        if self.transform:
            if hasattr(self.transform, '__call__'):
                # Albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
            else:
                # Torchvision
                image = self.transform(image)
        
        return image, label

def main():
    """
    Main entry point for breed preprocessing.
    
    Initializes the preprocessor and runs the complete pipeline.
    
    Returns:
        dict: Complete preprocessing results.
    """
    yesdog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\YESDOG"
    output_path = "./breed_processed_data"
    
    preprocessor = BreedDatasetPreprocessor(yesdog_path, output_path)
    results = preprocessor.run_complete_preprocessing(target_samples_per_class=200)
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n🎉 Ready for training!")
