#!/usr/bin/env python3
"""
Dataset Balancing Module for Dog Breed Classification
======================================================

This module provides tools for balancing the dog breed dataset through:

- Data augmentation (various image transformation techniques)
- Undersampling of over-represented classes
- Intelligent balancing strategies to reach target images per class

Supported augmentation techniques:
- Horizontal flip
- Rotation (+/- 15 degrees)
- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Center crop and resize

The module creates backups before modifications and generates
detailed reports of the balancing process.

Author: Dog Breed Classifier Team
Date: 2024
"""

import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import json
from pathlib import Path

class DatasetBalancer:
    """
    Dataset balancer for achieving uniform class distribution.
    
    This class handles both undersampling of over-represented breeds
    and oversampling through data augmentation for under-represented
    breeds to achieve a target number of images per class.
    
    Attributes:
        dataset_dir (str): Path to the dataset directory.
        target_images_per_class (int): Target number of images per breed.
        backup_dir (str): Path where original dataset backup is stored.
    """
    
    def __init__(self, dataset_dir, target_images_per_class=161):
        """
        Initialize the dataset balancer.
        
        Args:
            dataset_dir (str): Path to the dataset directory to balance.
            target_images_per_class (int): Target number of images per class.
                                          Default is 161.
        """
        self.dataset_dir = dataset_dir
        self.target_images_per_class = target_images_per_class
        self.backup_dir = f"{dataset_dir}_backup"
        
    def create_backup(self):
        """
        Create a backup of the original dataset.
        
        Creates a complete copy of the dataset directory before
        any modifications are applied.
        """
        if os.path.exists(self.backup_dir):
            print(f"⚠️ Backup already exists at: {self.backup_dir}")
            return
            
        print(f"💾 Creating backup at: {self.backup_dir}")
        shutil.copytree(self.dataset_dir, self.backup_dir)
        print("✅ Backup created successfully")
    
    def augment_image(self, image_path, output_path, augmentation_type):
        """
        Apply a specific augmentation technique to an image.
        
        Args:
            image_path (str): Path to the source image.
            output_path (str): Path where augmented image will be saved.
            augmentation_type (str): Type of augmentation to apply.
                Options: 'flip_horizontal', 'rotate_15', 'rotate_-15',
                        'brightness_up', 'brightness_down', 'contrast_up',
                        'contrast_down', 'saturation_up', 'saturation_down',
                        'crop_center'
        
        Returns:
            bool: True if augmentation was successful, False otherwise.
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                if augmentation_type == 'flip_horizontal':
                    augmented = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                elif augmentation_type == 'rotate_15':
                    augmented = img.rotate(15, expand=True, fillcolor='white')
                    
                elif augmentation_type == 'rotate_-15':
                    augmented = img.rotate(-15, expand=True, fillcolor='white')
                    
                elif augmentation_type == 'brightness_up':
                    enhancer = ImageEnhance.Brightness(img)
                    augmented = enhancer.enhance(1.3)
                    
                elif augmentation_type == 'brightness_down':
                    enhancer = ImageEnhance.Brightness(img)
                    augmented = enhancer.enhance(0.7)
                    
                elif augmentation_type == 'contrast_up':
                    enhancer = ImageEnhance.Contrast(img)
                    augmented = enhancer.enhance(1.3)
                    
                elif augmentation_type == 'contrast_down':
                    enhancer = ImageEnhance.Contrast(img)
                    augmented = enhancer.enhance(0.7)
                    
                elif augmentation_type == 'saturation_up':
                    enhancer = ImageEnhance.Color(img)
                    augmented = enhancer.enhance(1.3)
                    
                elif augmentation_type == 'saturation_down':
                    enhancer = ImageEnhance.Color(img)
                    augmented = enhancer.enhance(0.7)
                    
                elif augmentation_type == 'crop_center':
                    width, height = img.size
                    crop_size = min(width, height) * 0.8
                    left = (width - crop_size) / 2
                    top = (height - crop_size) / 2
                    right = (width + crop_size) / 2
                    bottom = (height + crop_size) / 2
                    augmented = img.crop((left, top, right, bottom))
                    augmented = augmented.resize((width, height))
                    
                else:
                    augmented = img  # Without cambios
                
                # Save augmented image
                augmented.save(output_path, 'JPEG', quality=90)
                return True
                
        except Exception as e:
            print(f"❌ Error augmenting {image_path}: {e}")
            return False
    
    def balance_breed(self, breed_name, current_count):
        """
        Balance a single breed to the target image count.
        
        Either reduces images through random selection or increases
        through data augmentation depending on current count.
        
        Args:
            breed_name (str): Name of the breed to balance.
            current_count (int): Current number of images for this breed.
        """
        breed_dir = os.path.join(self.dataset_dir, breed_name)
        
        if current_count > self.target_images_per_class:
            # Reduce images
            needed_reduction = current_count - self.target_images_per_class
            self._reduce_images(breed_dir, needed_reduction)
            print(f"   📉 {breed_name}: {current_count} → {self.target_images_per_class} (-{needed_reduction})")
            
        elif current_count < self.target_images_per_class:
            # Increase images with data augmentation
            needed_augmentation = self.target_images_per_class - current_count
            self._augment_images(breed_dir, needed_augmentation)
            print(f"   📈 {breed_name}: {current_count} → {self.target_images_per_class} (+{needed_augmentation})")
        
        else:
            print(f"   ✅ {breed_name}: {current_count} (already balanced)")
    
    def _reduce_images(self, breed_dir, reduction_needed):
        """
        Reduce images by random selection.
        
        Maintains the best quality images by randomly removing
        the specified number of images.
        
        Args:
            breed_dir (str): Path to the breed directory.
            reduction_needed (int): Number of images to remove.
        """
        image_files = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly select images to remove
        # In a more sophisticated implementation, quality metrics could be used
        to_remove = random.sample(image_files, reduction_needed)
        
        for img_file in to_remove:
            img_path = os.path.join(breed_dir, img_file)
            os.remove(img_path)
    
    def _augment_images(self, breed_dir, augmentation_needed):
        """
        Augment images to reach the target count.
        
        Applies various augmentation techniques to existing images
        to generate new training samples.
        
        Args:
            breed_dir (str): Path to the breed directory.
            augmentation_needed (int): Number of new images to generate.
        
        Returns:
            int: Number of images successfully augmented.
        """
        original_files = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Available augmentation types
        augmentation_types = [
            'flip_horizontal',
            'rotate_15',
            'rotate_-15', 
            'brightness_up',
            'brightness_down',
            'contrast_up',
            'contrast_down',
            'saturation_up',
            'saturation_down',
            'crop_center'
        ]
        
        augmented_count = 0
        attempts = 0
        max_attempts = augmentation_needed * 3  # Avoid infinite loop
        
        while augmented_count < augmentation_needed and attempts < max_attempts:
            # Select random base image
            base_image = random.choice(original_files)
            base_path = os.path.join(breed_dir, base_image)
            
            # Select random augmentation type
            aug_type = random.choice(augmentation_types)
            
            # Generate output filename
            base_name, ext = os.path.splitext(base_image)
            aug_filename = f"{base_name}_aug_{aug_type}_{augmented_count:03d}.jpg"
            aug_path = os.path.join(breed_dir, aug_filename)
            
            # Apply augmentation
            if self.augment_image(base_path, aug_path, aug_type):
                augmented_count += 1
            
            attempts += 1
        
        return augmented_count
    
    def balance_full_dataset(self):
        """
        Balance the entire dataset to target images per class.
        
        Processes all breeds, applies appropriate balancing strategy
        (undersample or oversample), and generates a final report.
        
        Returns:
            dict: Final report containing counts and statistics.
        """
        print("🔧 AUTOMATIC DATASET BALANCING")
        print("=" * 50)
        
        # Load balance report
        with open('detailed_balance_report.json', 'r') as f:
            report = json.load(f)
        
        breed_counts = report['analysis']['breed_counts']
        
        print(f"📊 Target: {self.target_images_per_class} images per breed")
        print(f"📁 Processing {len(breed_counts)} breeds...")
        
        # Create backup
        self.create_backup()
        
        # Process each breed
        for breed_name, current_count in breed_counts.items():
            self.balance_breed(breed_name, current_count)
        
        # Verify final result
        print(f"\n🔍 VERIFYING RESULT...")
        final_counts = {}
        total_final = 0
        
        for breed_name in breed_counts.keys():
            breed_dir = os.path.join(self.dataset_dir, breed_name)
            final_count = len([f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            final_counts[breed_name] = final_count
            total_final += final_count
        
        # Calculate statistics
        final_mean = np.mean(list(final_counts.values()))
        final_std = np.std(list(final_counts.values()))
        final_cv = final_std / final_mean
        
        print(f"\n📊 FINAL RESULT:")
        print(f"   Total images: {total_final:,}")
        print(f"   Average per breed: {final_mean:.1f}")
        print(f"   Standard deviation: {final_std:.1f}")
        print(f"   Coefficient of variation: {final_cv:.3f}")
        
        if final_cv < 0.05:
            print("   🟢 DATASET PERFECTLY BALANCED")
        elif final_cv < 0.1:
            print("   🟢 DATASET WELL BALANCED")
        else:
            print("   🟡 DATASET STILL NEEDS ADJUSTMENTS"))
        
        # Save reporte final
        final_report = {
            'balancing_target': self.target_images_per_class,
            'final_counts': final_counts,
            'final_stats': {
                'total_images': total_final,
                'mean': final_mean,
                'std': final_std,
                'cv': final_cv
            }
        }
        
        with open('balancing_final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n💾 Final report saved to: balancing_final_report.json")
        print(f"💾 Original backup at: {self.backup_dir}")
        
        return final_report

def main():
    """
    Main entry point for dataset balancing.
    
    Initializes the balancer with configuration and runs
    the complete balancing process.
    """
    
    # Check prerequisites
    if not os.path.exists('detailed_balance_report.json'):
        print("❌ First run detailed_balance_analysis.py")
        return
    
    # Configuration
    dataset_dir = "breed_processed_data/train"
    target_per_class = 161  # Median value from analysis
    
    # Create balancer
    balancer = DatasetBalancer(dataset_dir, target_per_class)
    
    # Execute balancing
    result = balancer.balance_full_dataset()
    
    print(f"\n✅ BALANCING COMPLETED")
    print(f"🚀 Ready to retrain the model with balanced dataset")

if __name__ == "__main__":
    main()