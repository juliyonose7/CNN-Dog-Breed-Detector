#!/usr/bin/env python3
"""
Targeted Data Augmentation Module for Dog Breed Classification.

This module implements targeted data augmentation strategies specifically designed
to address class imbalance and improve classification accuracy for problematic
dog breeds. It uses adaptive augmentation intensity based on class performance
metrics and generates balanced datasets through intelligent image transformation.

Key Features:
    - Automatic identification of underperforming classes from evaluation reports
    - Multi-level augmentation strategies (Critical, High, Medium, Normal)
    - Balanced dataset generation with configurable target counts
    - Visual reporting and analysis of augmentation results
    - Integration with albumentations library for advanced transformations

Author: AI System
Date: 2024
"""

import os
import json
import numpy as np
import cv2
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import random


class TargetedDataAugmenter:
    """
    Targeted data augmentation system for balancing dog breed datasets.
    
    This class implements intelligent data augmentation that adapts its intensity
    based on class performance metrics. Classes with lower accuracy receive more
    aggressive augmentation to improve their representation in the training data.
    
    Attributes:
        workspace_path (Path): Root path of the workspace.
        datasets_path (Path): Path to the datasets directory.
        yesdog_path (Path): Path to the YESDOG dataset.
        output_path (Path): Path for the balanced augmented dataset output.
        problematic_classes (list): List of tuples (breed_name, severity_level).
        class_accuracies (dict): Dictionary mapping breed names to their accuracy scores.
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize the TargetedDataAugmenter.
        
        Args:
            workspace_path (str): Absolute path to the workspace directory.
        """
        self.workspace_path = Path(workspace_path)
        self.datasets_path = self.workspace_path / "DATASETS"
        self.yesdog_path = self.datasets_path / "YESDOG"
        self.output_path = self.workspace_path / "BALANCED_AUGMENTED_DATASET"
        
        # Load classes with low accuracy for targeted improvement
        self.load_problematic_classes()
        
        # Configure augmentation pipelines for each severity level
        self.setup_augmentation_strategies()
        
    def load_problematic_classes(self):
        """
        Load and categorize classes with suboptimal accuracy from evaluation reports.
        
        Reads the class evaluation report and categorizes breeds by their accuracy:
            - CRITICAL: accuracy < 60%
            - HIGH: accuracy 60-70%
            - MEDIUM: accuracy 70-80%
        
        Falls back to known problematic breeds if no evaluation report exists.
        """
        eval_file = self.workspace_path / "complete_class_evaluation_report.json"
        
        self.problematic_classes = []
        self.class_accuracies = {}
        
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            class_details = results.get('class_details', {})
            for breed, details in class_details.items():
                accuracy = details['accuracy']
                self.class_accuracies[breed] = accuracy
                
                # Classify by problem severity level
                if accuracy < 0.60:
                    self.problematic_classes.append((breed, 'CRITICAL'))
                elif accuracy < 0.70:
                    self.problematic_classes.append((breed, 'HIGH'))
                elif accuracy < 0.80:
                    self.problematic_classes.append((breed, 'MEDIUM'))
        else:
            # Use hardcoded list of known problematic breeds if no report exists
            known_problematic = [
                ('Lhasa', 0.536),
                ('cairn', 0.586), 
                ('Siberian_husky', 0.621),
                ('whippet', 0.643),
                ('Australian_terrier', 0.690),
                ('Norfolk_terrier', 0.692),
                ('giant_schnauzer', 0.667),
                ('soft-coated_wheaten_terrier', 0.659)
            ]
            
            for breed, acc in known_problematic:
                self.class_accuracies[breed] = acc
                if acc < 0.60:
                    self.problematic_classes.append((breed, 'CRITICAL'))
                elif acc < 0.70:
                    self.problematic_classes.append((breed, 'HIGH'))
                else:
                    self.problematic_classes.append((breed, 'MEDIUM'))
        
        print(f" Problematic classes identified: {len(self.problematic_classes)}")
        for breed, level in self.problematic_classes:
            print(f"    {breed}: {level} (acc: {self.class_accuracies.get(breed, 0.0):.3f})")
    
    def setup_augmentation_strategies(self):
        """
        Configure augmentation pipelines for each severity level.
        
        Creates four augmentation strategies with varying intensities:
            - critical_augmentation: Aggressive transformations for worst performers
            - high_augmentation: Moderate-aggressive transformations
            - medium_augmentation: Conservative transformations
            - normal_augmentation: Light transformations for well-performing classes
        """
        
        # CRITICAL level: Most aggressive augmentation for worst performing classes
        self.critical_augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=0.3, p=0.5)
            ], p=0.6),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
            ], p=0.8),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ImageCompression(quality_lower=85, p=0.5)
            ], p=0.6),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.4)
        ])
        
        # HIGH level: Moderately aggressive augmentation
        self.high_augmentation = A.Compose([
            A.RandomRotate90(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.7),
            A.OneOf([
                A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.4),
                A.GridDistortion(p=0.4),
                A.OpticalDistortion(distort_limit=0.2, p=0.4)
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.4)
            ], p=0.6),
            A.OneOf([
                A.Blur(blur_limit=2, p=0.4),
                A.MotionBlur(blur_limit=2, p=0.4),
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.4)
            ], p=0.4),
            A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.3)
        ])
        
        # MEDIUM level: Conservative augmentation
        self.medium_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3)
            ], p=0.4),
            A.OneOf([
                A.Blur(blur_limit=1, p=0.3),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3)
            ], p=0.2),
            A.CoarseDropout(max_holes=2, max_height=16, max_width=16, p=0.2)
        ])
        
        # NORMAL level: Light augmentation for well-performing classes
        self.normal_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)
        ])
    
    def analyze_current_distribution(self):
        """
        Analyze the current image distribution across all breed classes.
        
        Calculates statistics about the dataset including mean, standard deviation,
        min/max counts, and identifies underrepresented classes that need augmentation.
        
        Returns:
            dict: Distribution analysis containing:
                - class_counts: Image count per breed
                - target_count: Target images per class (max observed)
                - underrepresented: List of classes needing augmentation
                - stats: Statistical summary of the distribution
            None: If the YESDOG directory doesn't exist
        """
        print(f"\n ANALYZING CURRENT DISTRIBUTION")
        print("="*50)
        
        class_counts = {}
        total_images = 0
        
        if not self.yesdog_path.exists():
            print(f" YESDOG directory not found: {self.yesdog_path}")
            return None
        
        for breed_dir in self.yesdog_path.iterdir():
            if breed_dir.is_dir():
                breed_name = breed_dir.name
                # Count images (jpg, jpeg, png)
                image_files = list(breed_dir.glob("*.jpg")) + \
                             list(breed_dir.glob("*.jpeg")) + \
                             list(breed_dir.glob("*.png")) + \
                             list(breed_dir.glob("*.JPEG"))
                
                count = len(image_files)
                class_counts[breed_name] = count
                total_images += count
        
        # Calculate statistics
        counts = list(class_counts.values())
        if counts:
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            min_count = min(counts)
            max_count = max(counts)
            
            print(f" CURRENT STATISTICS:")
            print(f"   Total classes: {len(class_counts)}")
            print(f"   Total images: {total_images:,}")
            print(f"   Average per class: {mean_count:.1f}")
            print(f"   Standard deviation: {std_count:.1f}")
            print(f"   Range: {min_count} - {max_count}")
            print(f"   Coefficient of variation: {std_count/mean_count:.3f}")
            
            # Identify classes with few images
            target_count = max_count  # Target: match the class with most images
            underrepresented = [(breed, count, target_count-count) 
                              for breed, count in class_counts.items() 
                              if count < target_count]
            
            underrepresented.sort(key=lambda x: x[1])  # Sort by current count
            
            print(f"\n BALANCING TARGET: {target_count} images per class")
            print(f" Classes needing augmentation: {len(underrepresented)}")
            
            if underrepresented:
                print(f"\n TOP 10 MOST IMBALANCED CLASSES:")
                for i, (breed, current, needed) in enumerate(underrepresented[:10], 1):
                    problem_level = "NORMAL"
                    for prob_breed, level in self.problematic_classes:
                        if prob_breed == breed:
                            problem_level = level
                            break
                    
                    print(f"   {i:2d}. {breed:25} | Current: {current:3d} | Needs: +{needed:3d} | {problem_level}")
            
            return {
                'class_counts': class_counts,
                'target_count': target_count,
                'underrepresented': underrepresented,
                'stats': {
                    'mean': mean_count,
                    'std': std_count,
                    'min': min_count,
                    'max': max_count,
                    'total': total_images,
                    'classes': len(class_counts)
                }
            }
        
        return None
    
    def get_augmentation_strategy(self, breed_name, current_count, needed_count):
        """
        Determine the appropriate augmentation strategy for a breed.
        
        Selects augmentation intensity based on the breed's classification accuracy
        and the proportion of images needed relative to current count.
        
        Args:
            breed_name (str): Name of the breed to augment.
            current_count (int): Current number of images for this breed.
            needed_count (int): Number of additional images needed.
            
        Returns:
            tuple: (augmentation_pipeline, severity_level, variations_per_image)
        """
        
        # Determine problem severity level
        problem_level = "NORMAL"
        for prob_breed, level in self.problematic_classes:
            if prob_breed == breed_name:
                problem_level = level
                break
        
        # Determine augmentation intensity based on:
        # 1. Problem severity (accuracy)
        # 2. Number of missing images
        
        shortage_ratio = needed_count / max(current_count, 1)
        
        if problem_level == "CRITICAL" or shortage_ratio > 3:
            return self.critical_augmentation, "CRITICAL", 6  # 6 variations per image
        elif problem_level == "HIGH" or shortage_ratio > 2:
            return self.high_augmentation, "HIGH", 4
        elif problem_level == "MEDIUM" or shortage_ratio > 1.5:
            return self.medium_augmentation, "MEDIUM", 3
        else:
            return self.normal_augmentation, "NORMAL", 2
    
    def augment_breed_images(self, breed_name, breed_path, target_count, current_count):
        """
        Perform data augmentation for a specific breed to reach target count.
        
        Copies original images and generates augmented versions using the
        appropriate augmentation strategy based on breed performance.
        
        Args:
            breed_name (str): Name of the breed to augment.
            breed_path (Path): Path to the breed's image directory.
            target_count (int): Target number of images to achieve.
            current_count (int): Current number of images available.
            
        Returns:
            int: Number of augmented images generated.
        """
        
        needed_count = target_count - current_count
        if needed_count <= 0:
            return 0
        
        # Get augmentation strategy
        augmentation, strategy_level, variations_per_image = self.get_augmentation_strategy(
            breed_name, current_count, needed_count
        )
        
        print(f"    {breed_name} | Current: {current_count} | Target: {target_count} | Strategy: {strategy_level}")
        
        # Create output directory
        output_breed_path = self.output_path / breed_name
        output_breed_path.mkdir(parents=True, exist_ok=True)
        
        # Copy original images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG']
        original_images = []
        
        for ext in image_extensions:
            original_images.extend(list(breed_path.glob(ext)))
        
        copied_count = 0
        for img_path in original_images:
            try:
                shutil.copy2(img_path, output_breed_path / img_path.name)
                copied_count += 1
            except Exception as e:
                print(f"       Error copying {img_path.name}: {e}")
        
        # Generate augmented images
        generated_count = 0
        images_to_augment = original_images * ((needed_count // len(original_images)) + 1)
        random.shuffle(images_to_augment)
        
        for i, img_path in enumerate(images_to_augment[:needed_count]):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation
                augmented = augmentation(image=image)
                aug_image = augmented['image']
                
                # Save augmented image
                aug_filename = f"{img_path.stem}_aug_{i:04d}{img_path.suffix}"
                aug_path = output_breed_path / aug_filename
                
                aug_image_pil = Image.fromarray(aug_image)
                aug_image_pil.save(aug_path, quality=95)
                
                generated_count += 1
                
                if generated_count % 50 == 0:
                    print(f"       Generated {generated_count}/{needed_count} images...")
                    
            except Exception as e:
                print(f"       Error augmenting {img_path.name}: {e}")
                continue
        
        final_count = copied_count + generated_count
        print(f"       Completed: {copied_count} original + {generated_count} augmented = {final_count} total")
        
        return generated_count
    
    def create_balanced_dataset(self):
        """
        Create a balanced dataset using targeted data augmentation.
        
        Orchestrates the complete pipeline of analyzing distribution,
        augmenting underrepresented classes, and generating the output dataset.
        
        Returns:
            dict: Results containing output path, target count, statistics, etc.
            None: If the process fails.
        """
        print(f"\n STARTING BALANCED DATASET CREATION")
        print("="*70)
        
        # Analyze current image distribution
        distribution_data = self.analyze_current_distribution()
        if not distribution_data:
            print(" Could not analyze current distribution")
            return None
        
        class_counts = distribution_data['class_counts']
        target_count = distribution_data['target_count']
        underrepresented = distribution_data['underrepresented']
        
        # Create output directory
        if self.output_path.exists():
            print(f" Cleaning existing directory...")
            shutil.rmtree(self.output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each class
        print(f"\n PROCESSING {len(class_counts)} CLASSES...")
        print("="*70)
        
        total_generated = 0
        successful_classes = 0
        
        for breed_name, current_count in class_counts.items():
            breed_path = self.yesdog_path / breed_name
            
            if not breed_path.exists():
                print(f" Directory not found: {breed_name}")
                continue
            
            try:
                generated = self.augment_breed_images(
                    breed_name, breed_path, target_count, current_count
                )
                total_generated += generated
                successful_classes += 1
                
            except Exception as e:
                print(f" Error processing {breed_name}: {e}")
                continue
        
        # Verify final result
        final_distribution = self.verify_balanced_dataset()
        
        # Summary
        print(f"\n BALANCED DATASET CREATED SUCCESSFULLY")
        print("="*70)
        print(f"    Location: {self.output_path}")
        print(f"    Classes processed: {successful_classes}/{len(class_counts)}")
        print(f"    Images generated: {total_generated:,}")
        print(f"    Target per class: {target_count}"))
        
        if final_distribution:
            print(f"    Balance achieved: {final_distribution['balanced_classes']}/{final_distribution['total_classes']} classes")
            print(f"    Final total: {final_distribution['total_images']:,} images")
        
        return {
            'output_path': str(self.output_path),
            'target_count': target_count,
            'generated_images': total_generated,
            'successful_classes': successful_classes,
            'total_classes': len(class_counts),
            'final_distribution': final_distribution
        }
    
    def verify_balanced_dataset(self):
        """
        Verify the balanced dataset was created correctly.
        
        Counts images in the output directory and calculates balance metrics.
        
        Returns:
            dict: Verification results with class counts and balance statistics.
            None: If the output directory doesn't exist.
        """
        if not self.output_path.exists():
            return None
        
        class_counts = {}
        for breed_dir in self.output_path.iterdir():
            if breed_dir.is_dir():
                # Count all images
                image_files = list(breed_dir.glob("*.jpg")) + \
                             list(breed_dir.glob("*.jpeg")) + \
                             list(breed_dir.glob("*.png")) + \
                             list(breed_dir.glob("*.JPEG"))
                
                class_counts[breed_dir.name] = len(image_files)
        
        if not class_counts:
            return None
        
        counts = list(class_counts.values())
        target_count = max(counts)
        balanced_classes = sum(1 for count in counts if count >= target_count * 0.95)  # 95% of the target
        
        return {
            'class_counts': class_counts,
            'target_count': target_count,
            'balanced_classes': balanced_classes,
            'total_classes': len(class_counts),
            'total_images': sum(counts),
            'mean_count': np.mean(counts),
            'std_count': np.std(counts)
        }
    
    def create_visualization_report(self, results):
        """
        Create a visual report of the augmentation process.
        
        Generates plots showing distribution before/after, problem class improvements,
        and overall statistics. Saves results as PNG and JSON files.
        
        Args:
            results (dict): Results from create_balanced_dataset().
        """
        if not results or not results.get('final_distribution'):
            print(" No data available for visualization")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(' TARGETED DATA AUGMENTATION REPORT', fontsize=16, fontweight='bold')
        
        final_dist = results['final_distribution']
        class_counts = final_dist['class_counts']
        
        # 1. Final distribution (Top 20 classes)
        breeds = list(class_counts.keys())[:20]  # Top 20 for readability
        counts = [class_counts[breed] for breed in breeds]
        
        bars1 = ax1.bar(range(len(breeds)), counts, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.axhline(y=final_dist['target_count'], color='red', linestyle='--', 
                   label=f"Objetivo: {final_dist['target_count']}")
        ax1.set_xticks(range(len(breeds)))
        ax1.set_xticklabels([breed.replace('_', ' ') for breed in breeds], 
                           rotation=45, ha='right', fontsize=8)
        ax1.set_title(' Final Distribution (Top 20 Classes)'))
        ax1.set_ylabel('Number of Images')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution histogram
        all_counts = list(class_counts.values())
        ax2.hist(all_counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.axvline(final_dist['mean_count'], color='red', linestyle='-', 
                   label=f"Mean: {final_dist['mean_count']:.0f}")
        ax2.axvline(final_dist['target_count'], color='orange', linestyle='--',
                   label=f"Objetivo: {final_dist['target_count']}")
        ax2.set_xlabel('Images per Class')
        ax2.set_ylabel('Number of Classes')
        ax2.set_title(' Image Distribution per Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Problematic classes improvement
        problematic_breeds = [breed for breed, level in self.problematic_classes]
        problematic_counts = [class_counts.get(breed, 0) for breed in problematic_breeds]
        problematic_levels = [level for breed, level in self.problematic_classes]
        
        colors_map = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow'}
        colors3 = [colors_map.get(level, 'gray') for level in problematic_levels]
        
        bars3 = ax3.bar(range(len(problematic_breeds)), problematic_counts, 
                       color=colors3, alpha=0.7, edgecolor='black')
        ax3.axhline(y=final_dist['target_count'], color='green', linestyle='--',
                   label=f"Objetivo: {final_dist['target_count']}")
        ax3.set_xticks(range(len(problematic_breeds)))
        ax3.set_xticklabels([breed.replace('_', ' ') for breed in problematic_breeds], 
                           rotation=45, ha='right', fontsize=9)
        ax3.set_title(' Improved Problematic Classes')
        ax3.set_ylabel('Final Images')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Improvement statistics
        stats_labels = ['Classes\nProcessed', 'Images\nGenerated', 'Balance\nAchieved %']
        stats_values = [
            results['successful_classes'],
            results['generated_images'],
            (final_dist['balanced_classes'] / final_dist['total_classes']) * 100
        ]
        
        bars4 = ax4.bar(stats_labels, stats_values, 
                       color=['lightblue', 'lightcoral', 'lightgreen'], 
                       alpha=0.7, edgecolor='black')
        ax4.set_title(' Improvement Statistics')
        ax4.set_ylabel('Value')
        
        # Add values on bars
        for bar, value in zip(bars4, stats_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(stats_values)*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('targeted_augmentation_report.png', dpi=300, bbox_inches='tight')
        print(" Visual report saved: targeted_augmentation_report.png")
        
        # Save JSON report
        report_data = {
            'timestamp': str(np.datetime64('now')),
            'results': results,
            'problematic_classes': dict(self.problematic_classes),
            'class_accuracies': self.class_accuracies
        }
        
        with open('targeted_augmentation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(" JSON report saved: targeted_augmentation_report.json")
    
    def run_complete_augmentation(self):
        """
        Execute the complete targeted augmentation pipeline.
        
        Orchestrates dataset creation and report generation.
        
        Returns:
            dict: Complete results from the augmentation process.
            None: If the process fails.
        """
        print("" * 70)
        print(" TARGETED DATA AUGMENTATION FOR PROBLEMATIC BREEDS")
        print("" * 70)
        
        try:
            # Create dataset balanced
            results = self.create_balanced_dataset()
            
            if results:
                # Create visual report
                self.create_visualization_report(results)
                
                print(f"\n PROCESS COMPLETED SUCCESSFULLY")
                print(f"    Balanced dataset: {results['output_path']}")
                print(f"    Images generated: {results['generated_images']:,}")
                print(f"    Balanced classes: {results['successful_classes']}/{results['total_classes']}"))
                
                return results
            else:
                print(" Error in augmentation process")
                return None
                
        except Exception as e:
            print(f" Error in complete process: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point for targeted data augmentation."""
    workspace_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
    
    augmenter = TargetedDataAugmenter(workspace_path)
    results = augmenter.run_complete_augmentation()
    
    return results

if __name__ == "__main__":
    results = main()