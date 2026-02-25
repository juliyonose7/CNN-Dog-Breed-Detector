"""
Dataset Analyzer for Binary Dog vs Non-Dog Classification.

This module provides specialized analysis tools for examining the structure,
quality, and distribution of images in the training dataset. It is optimized
for AMD hardware configurations.

Main Features:
    - Directory structure and class distribution analysis
    - Image statistics: resolution, format, quality metrics
    - Class imbalance detection
    - File integrity validation
    - Visual report and metrics generation
    - Potential data issues identification
    - Dataset improvement recommendations

Components:
    - DatasetAnalyzer: Main class for comprehensive dataset analysis

Usage:
    from data_analyzer import DatasetAnalyzer
    
    analyzer = DatasetAnalyzer("path/to/DATASETS")
    analyzer.run_complete_analysis()

Author: System IA
Date: 2024
"""

# System and file handling imports
import os                    # Operating system operations
import json                  # JSON data handling
from pathlib import Path     # Modern path handling
from collections import defaultdict, Counter  # Specialized data structures
import warnings              # Warning control
warnings.filterwarnings('ignore')  # Suppress non-critical warnings

# Image processing imports
import cv2                   # OpenCV computer vision
import numpy as np           # Numerical operations
from tqdm import tqdm        # Progress bars

# Data analysis and visualization imports
import pandas as pd          # DataFrame manipulation
import matplotlib.pyplot as plt  # Graphs and plots
import seaborn as sns        # Advanced statistical visualizations


class DatasetAnalyzer:
    """
    Comprehensive image dataset analyzer for evaluating data quality and structure.
    
    This class provides tools to analyze directory structure, image properties,
    class distribution, and data quality issues in datasets organized for
    binary (dog vs non-dog) classification tasks.
    
    Attributes:
        dataset_path (Path): Root directory path containing YESDOG and NODOG folders.
        yesdog_path (Path): Path to dog images subdirectory.
        nodog_path (Path): Path to non-dog images subdirectory.
        stats (dict): Dictionary storing all computed statistics.
        image_extensions (set): Set of supported image file extensions.
    
    Example:
        >>> analyzer = DatasetAnalyzer("/path/to/DATASETS")
        >>> analyzer.run_complete_analysis()
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the analyzer with the path to the main dataset directory.
        
        Args:
            dataset_path (str): Path to root directory containing YESDOG and NODOG folders.
        """
        self.dataset_path = Path(dataset_path)     # Main dataset path
        self.yesdog_path = self.dataset_path / "YESDOG"  # Dog images subdirectory
        self.nodog_path = self.dataset_path / "NODOG"    # Non-dog images subdirectory
        self.stats = {}                            # Dictionary for storing statistics
        
        # Supported image extensions for validation
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def analyze_dataset_structure(self):
        """
        Analyze the complete dataset structure and class distribution.
        
        This method examines all subdirectories in both YESDOG and NODOG folders,
        counting images and compiling distribution statistics for each breed
        and category.
        
        Returns:
            None: Results are stored in self.stats dictionary.
        
        Updates:
            self.stats with keys:
                - 'dog_breeds': List of breed info dicts with names and counts
                - 'nodog_categories': List of category info dicts
                - 'total_dog_images': Total count of dog images
                - 'total_nodog_images': Total count of non-dog images
                - 'total_images': Combined total image count
                - 'class_balance': Balance metrics between classes
        """
        print("🔍 Analyzing dataset structure...")
        
        # Analysis of YESDOG category with all dog breeds
        dog_breeds = []           # List to store each breed's info
        dog_image_count = 0       # Total dog image counter
        
        # Iterate over each breed subdirectory in YESDOG
        for breed_folder in self.yesdog_path.iterdir():
            if breed_folder.is_dir():  # Verify it's a valid directory
                breed_name = breed_folder.name  # Breed name
                images = self._count_images_in_folder(breed_folder)  # Count images
                
                # Store breed information
                dog_breeds.append({
                    'breed': breed_name,
                    'folder': str(breed_folder),
                    'image_count': images
                })
                dog_image_count += images  # Add to total
        
        # Analysis of NODOG category with non-dog objects
        nodog_categories = []     # List to store each category's info
        nodog_image_count = 0     # Total non-dog image counter
        
        # Iterate over each category subdirectory in NODOG
        for category_folder in self.nodog_path.iterdir():
            if category_folder.is_dir():  # Verify it's a valid directory
                category_name = category_folder.name  # Category name
                images = self._count_images_in_folder(category_folder)  # Count images
                
                # Store category information
                nodog_categories.append({
                    'category': category_name,
                    'folder': str(category_folder),
                    'image_count': images
                })
                nodog_image_count += images  # Add to total
        
        # Store complete statistics in main object
        self.stats['dog_breeds'] = dog_breeds               # List of breeds with counts
        self.stats['nodog_categories'] = nodog_categories   # List of categories with counts
        self.stats['total_dog_images'] = dog_image_count    # Total dog images
        self.stats['total_nodog_images'] = nodog_image_count # Total non-dog images
        self.stats['total_images'] = dog_image_count + nodog_image_count  # Overall total
        
        # Calculate balance metrics between main classes
        self.stats['class_balance'] = {
            'dogs': dog_image_count,      # Absolute dog count
            'no_dogs': nodog_image_count, # Absolute non-dog count
            'ratio': dog_image_count / max(nodog_image_count, 1)  # Ratio avoiding division by zero
        }
        
        # Print executive summary of the analysis
        print(f"✅ Analysis completed:")
        print(f"   - Dog breeds: {len(dog_breeds)}")
        print(f"   - Non-dog categories: {len(nodog_categories)}")
        print(f"   - Total dog images: {dog_image_count:,}")
        print(f"   - Total non-dog images: {nodog_image_count:,}")
        print(f"   - Dog/non-dog ratio: {self.stats['class_balance']['ratio']:.2f}")
        
    def _count_images_in_folder(self, folder_path: Path) -> int:
        """
        Count valid images in a specific directory.
        
        Filters only files with recognized image extensions.
        
        Args:
            folder_path (Path): Path to the directory to analyze.
        
        Returns:
            int: Number of valid images found.
        """
        count = 0  # Initialize counter
        
        # Examine each file in the directory
        for file in folder_path.iterdir():
            # Verify it's a file with valid image extension
            if file.is_file() and file.suffix.lower() in self.image_extensions:
                count += 1  # Increment counter if valid image
                    
        return count  # Return total images found
    
    def analyze_image_properties(self, sample_size: int = 1000):
        """
        Analyze technical properties of images through statistical sampling.
        
        Examines dimensions, quality, and visual characteristics of images
        in the dataset using a representative sample.
        
        Args:
            sample_size (int): Total number of images to sample from dataset.
                Defaults to 1000.
        
        Features:
            - Dimension distribution: width, height, color channels
            - File size statistics in KB
            - Corrupted or unreadable file detection
            - Aspect ratio analysis
        
        Returns:
            None: Results stored in self.stats['image_properties'].
        """
        print(f"📊 Analyzing image properties on sample of {sample_size}...")
        
        # Dictionary to collect all measured properties
        image_properties = {
            'widths': [],         # List of widths in pixels
            'heights': [],        # List of heights in pixels
            'channels': [],       # List of color channel counts
            'file_sizes': [],     # List of file sizes in KB
            'corrupted': [],      # List of corrupted/unreadable files
            'aspect_ratios': []   # List of width/height ratios
        }
        
        # Get balanced samples from both main classes
        dog_samples = self._sample_images_from_class('dog', sample_size // 2)      # Half dogs
        nodog_samples = self._sample_images_from_class('nodog', sample_size // 2)  # Half non-dogs
        
        # Combine all samples for unified analysis
        all_samples = dog_samples + nodog_samples
        
        # Process each individual image with progress bar
        for img_path, label in tqdm(all_samples, desc="Analyzing images"):
            try:
                # Load image using OpenCV for analysis
                img = cv2.imread(str(img_path))
                
                # Verify if image loaded correctly
                if img is None:
                    image_properties['corrupted'].append(str(img_path))  # Mark as corrupted
                    continue  # Skip to next file
                
                # Extract basic image dimensions
                h, w, c = img.shape  # Height, width, channels
                image_properties['heights'].append(h)      # Store height
                image_properties['widths'].append(w)       # Store width
                image_properties['channels'].append(c)     # Store channels
                image_properties['aspect_ratios'].append(w/h)  # Calculate and store ratio
                
                # Get file size in KB
                file_size = Path(img_path).stat().st_size / 1024  # Convert bytes to KB
                image_properties['file_sizes'].append(file_size)  # Store file size
                
            except Exception as e:
                # Handle any errors during image analysis
                image_properties['corrupted'].append(str(img_path))  # Mark as corrupted
        
        # Calculate complete descriptive statistics for properties
        self.stats['image_properties'] = {
            'width_stats': {               # Width statistics
                'mean': np.mean(image_properties['widths']),    # Average
                'std': np.std(image_properties['widths']),      # Standard deviation
                'min': np.min(image_properties['widths']),      # Minimum
                'max': np.max(image_properties['widths']),      # Maximum
                'median': np.median(image_properties['widths']) # Median
            },
            'height_stats': {              # Height statistics
                'mean': np.mean(image_properties['heights']),   # Average
                'std': np.std(image_properties['heights']),     # Standard deviation
                'min': np.min(image_properties['heights']),     # Minimum
                'max': np.max(image_properties['heights']),     # Maximum
                'median': np.median(image_properties['heights'])# Median
            },
            'aspect_ratio_stats': {        # Aspect ratio statistics
                'mean': np.mean(image_properties['aspect_ratios']),   # Average
                'std': np.std(image_properties['aspect_ratios']),     # Standard deviation
                'min': np.min(image_properties['aspect_ratios']),     # Most square
                'max': np.max(image_properties['aspect_ratios'])      # Most rectangular
            },
            'file_size_stats': {           # File size statistics
                'mean_kb': np.mean(image_properties['file_sizes']),   # Average in KB
                'median_kb': np.median(image_properties['file_sizes']),# Median in KB
                'min_kb': np.min(image_properties['file_sizes']),     # Minimum in KB
                'max_kb': np.max(image_properties['file_sizes'])      # Maximum in KB
            },
            'corrupted_count': len(image_properties['corrupted']),   # Total corrupted files
            'total_analyzed': len(all_samples)                       # Total images analyzed
        }
        
        # Print executive summary of property analysis
        print(f"✅ Properties analyzed:")
        print(f"   - Corrupted images: {len(image_properties['corrupted'])}")
        print(f"   - Average dimensions: {self.stats['image_properties']['width_stats']['mean']:.0f}x{self.stats['image_properties']['height_stats']['mean']:.0f}")
        print(f"   - Average size: {self.stats['image_properties']['file_size_stats']['mean_kb']:.1f} KB")
        
    def _sample_images_from_class(self, class_type: str, sample_size: int):
        """
        Sample images from a specific class proportionally.
        
        Distributes sampling evenly across subcategories to avoid bias.
        
        Args:
            class_type (str): Class to sample from ('dog' or 'nodog').
            sample_size (int): Total number of images to retrieve.
        
        Returns:
            list: List of tuples (image_path, class_label).
        
        Sampling Strategy:
            - Proportional sampling per subcategory to avoid bias
            - Equal distribution between breeds or categories
            - Random selection within each subcategory
        """
        images = []  # List to store sampled images
        
        # Determine directories based on requested class type
        if class_type == 'dog':
            # Get all dog breed folders
            folders = [breed['folder'] for breed in self.stats['dog_breeds']]
        else:
            # Get all non-dog category folders
            folders = [cat['folder'] for cat in self.stats['nodog_categories']]
        
        # Process each subcategory individually
        for folder in folders:
            folder_path = Path(folder)  # Convert to Path object
            folder_images = []          # Images found in this folder
            
            # Collect all valid images from current folder
            for file in folder_path.iterdir():
                # Verify it's a file with image extension
                if file.is_file() and file.suffix.lower() in self.image_extensions:
                    folder_images.append((file, class_type))  # Add image-label tuple
            
            # Perform proportional sampling if images are available
            if folder_images:
                # Calculate number of samples for this folder
                n_samples = min(len(folder_images), max(1, sample_size // len(folders)))
                
                # Select random indices without replacement
                sampled = np.random.choice(len(folder_images), 
                                         size=min(n_samples, len(folder_images)), 
                                         replace=False)  # Avoid duplicates
                
                # Add selected images to final list
                for idx in sampled:
                    images.append(folder_images[idx])
        
        return images  # Return complete list of sampled images
    
    def create_visualization_report(self):
        """
        Create comprehensive visualizations of the dataset analysis.
        
        Generates charts to understand data distribution and characteristics.
        
        Features:
            - Main class distribution pie chart
            - Top breeds and categories bar charts
            - Image property histograms
            - Correlation heatmaps
            - Exportable PNG format reports
        
        Returns:
            None: Saves visualization to 'dataset_analysis_report.png'.
        """
        print("📈 Creating visual report...")
        
        # Configure main figure with multiple organized subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 grid for 6 charts
        fig.suptitle('DOG vs NON-DOG Dataset Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Main class distribution with pie chart
        ax1 = axes[0, 0]
        classes = ['Dogs', 'Non-Dogs']                    # Class labels
        counts = [self.stats['total_dog_images'], self.stats['total_nodog_images']]  # Counts
        colors = ['#FF6B6B', '#4ECDC4']  # Distinctive colors
        
        # Create pie chart with automatic percentages
        wedges, texts, autotexts = ax1.pie(counts, labels=classes, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Class Distribution')  # Descriptive title
        
        # Plot 2: Top 10 dog breeds with most images available
        ax2 = axes[0, 1]
        dog_df = pd.DataFrame(self.stats['dog_breeds'])      # Convert to DataFrame
        top_breeds = dog_df.nlargest(10, 'image_count')     # Get top 10
        
        # Extract breed names without technical prefixes
        breed_names = [breed.split('-')[-1] for breed in top_breeds['breed']]
        
        # Create horizontal bar plot for better readability
        ax2.barh(breed_names, top_breeds['image_count'], color='#FF6B6B', alpha=0.7)
        ax2.set_title('Top 10 Breeds by Image Count')     # Descriptive title
        ax2.set_xlabel('Number of Images')            # X-axis label
        
        # Plot 3: Non-dog categories with highest representation
        ax3 = axes[0, 2]
        nodog_df = pd.DataFrame(self.stats['nodog_categories'])  # Convert to DataFrame
        
        # Clean category names by removing technical suffixes
        cat_names = [cat.replace('_final', '') for cat in nodog_df['category']]
        
        # Create vertical bar chart for non-dog categories
        ax3.bar(range(len(cat_names)), nodog_df['image_count'], color='#4ECDC4', alpha=0.7)
        ax3.set_title('Non-Dog Categories')         # Descriptive title
        ax3.set_xlabel('Category')                   # X-axis label
        ax3.set_ylabel('Number of Images')        # Y-axis label
        ax3.set_xticks(range(len(cat_names)))       # Label positions
        ax3.set_xticklabels(cat_names, rotation=45, ha='right')  # Rotated labels
        
        # Plots 4-6: Technical property analysis if available
        if 'image_properties' in self.stats:
            # Plot 4: Image dimension statistics with bar chart
            ax4 = axes[1, 0]
            width_stats = self.stats['image_properties']['width_stats']   # Width statistics
            height_stats = self.stats['image_properties']['height_stats'] # Height statistics
            
            # Prepare data for bar chart with standard deviation
            dimensions = ['Width', 'Height']                              # Labels
            means = [width_stats['mean'], height_stats['mean']]         # Averages
            stds = [width_stats['std'], height_stats['std']]            # Standard deviations
            
            # Create bar chart with error bars for standard deviation
            ax4.bar(dimensions, means, yerr=stds, capsize=5, 
                   color=['#95E1D3', '#F38BA8'], alpha=0.7)
            ax4.set_title('Average Image Dimensions')    # Title
            ax4.set_ylabel('Pixels')                          # Units
            
            # Plot 5: Aspect ratio statistics as formatted text
            ax5 = axes[1, 1]
            ar_stats = self.stats['image_properties']['aspect_ratio_stats']  # Ratio statistics
            
            # Show key metrics as formatted text
            ax5.text(0.1, 0.8, f"Average Aspect Ratio: {ar_stats['mean']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.6, f"Standard Deviation: {ar_stats['std']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.4, f"Range: {ar_stats['min']:.2f} - {ar_stats['max']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Aspect Ratio Statistics')  # Title
            ax5.axis('off')  # Hide axes for clean presentation
            
            # Plot 6: Dataset quality with valid vs corrupted proportion
            ax6 = axes[1, 2]
            total_analyzed = self.stats['image_properties']['total_analyzed']  # Total analyzed
            corrupted = self.stats['image_properties']['corrupted_count']     # Corrupted files
            valid = total_analyzed - corrupted                               # Valid files
            
            # Data for quality pie chart
            quality_data = ['Valid', 'Corrupted']         # Labels
            quality_counts = [valid, corrupted]            # Counts
            quality_colors = ['#90EE90', '#FFB6C1']  # Semantic colors
            
            # Create pie chart to show quality proportion
            ax6.pie(quality_counts, labels=quality_data, autopct='%1.1f%%', 
                   colors=quality_colors, startangle=90)
            ax6.set_title('Sample Image Quality')  # Title
        
        # Adjust spacing and save report as high-quality image
        plt.tight_layout()  # Optimize spacing between subplots
        plt.savefig(self.dataset_path / 'dataset_analysis_report.png', 
                   dpi=300, bbox_inches='tight')  # Export in high resolution
        plt.show()  # Display on screen
        
        # Confirm location of generated file
        print(f"✅ Report saved to: {self.dataset_path / 'dataset_analysis_report.png'}")
    
    def generate_recommendations(self):
        """
        Generate intelligent recommendations for optimizing model training.
        
        Analyzes dataset metrics to suggest best practices for model development.
        
        Features:
            - Class balance evaluation
            - Balancing technique recommendations
            - Optimized preprocessing suggestions
            - Model architecture recommendations
            - Hyperparameter configuration suggestions
        
        Returns:
            None: Prints recommendations to stdout.
        """
        print("\n💡 MODEL RECOMMENDATIONS:")
        print("="*50)
        
        # Analyze class balance and suggest corrections if needed
        ratio = self.stats['class_balance']['ratio']  # Dog/non-dog ratio
        
        # Check if significant imbalance exists between classes
        if ratio > 2 or ratio < 0.5:  # Critical imbalance threshold
            print(f"⚠️  CLASS IMBALANCE detected ratio: {ratio:.2f}")
            print("   → Use balancing techniques: oversampling, undersampling, or class weights")
        else:
            print(f"✅ Class balance acceptable ratio: {ratio:.2f}")
        
        # Evaluate total dataset size
        total = self.stats['total_images']  # Total available images
        
        # Determine if dataset is large enough for robust training
        if total < 10000:  # Recommended minimum threshold
            print(f"⚠️  Small dataset: {total:,} images")
            print("   → Use aggressive data augmentation")
            print("   → Consider transfer learning with pretrained models")
        else:
            print(f"✅ Adequate dataset size: {total:,} images")
        
        # Analyze technical properties if available
        if 'image_properties' in self.stats:
            # Evaluate file corruption rate
            corruption_rate = (self.stats['image_properties']['corrupted_count'] / 
                             self.stats['image_properties']['total_analyzed'])
            
            # Alert if corruption rate is concerning
            if corruption_rate > 0.01:  # More than 1% corrupted is concerning
                print(f"⚠️  High percentage of corrupted images: {corruption_rate*100:.1f}%")
                print("   → Implement robust image validation")
            
            # Extract average dimensions for preprocessing recommendations
            avg_width = self.stats['image_properties']['width_stats']['mean']
            avg_height = self.stats['image_properties']['height_stats']['mean']
            
            # Optimized preprocessing recommendations section
            print(f"\n📋 RECOMMENDED PREPROCESSING:")
            print(f"   • Resize to: 224x224 standard for transfer learning")
            print(f"   • Normalization: ImageNet stats [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]")
            print(f"   • Augmentation: rotation ±15°, horizontal flip, random crop")
            
        # Model and architecture recommendations section
        print(f"\n🎯 RECOMMENDED MODEL:")
        print(f"   • EfficientNet-B3 or ResNet-50 pretrained on ImageNet")
        print(f"   • Transfer learning: freeze initial layers, fine-tune last layers")
        print(f"   • Optimizer: AdamW with learning rate scheduling")
        print(f"   • Loss: BCEWithLogitsLoss with class weights if imbalanced")
        
    def save_analysis_report(self):
        """
        Save complete analysis report in JSON format.
        
        Exports all statistics and metrics collected during analysis.
        
        Features:
            - Serializes all stats dictionary contents
            - JSON formatting with readable indentation
            - Preserves Unicode characters for breed names
            - Converts non-serializable objects to strings
            - Generates persistent file for future reference
        
        Returns:
            None: Saves report to 'dataset_analysis_report.json'.
        """
        # Define report file path in dataset directory
        report_path = self.dataset_path / 'dataset_analysis_report.json'
        
        # Write JSON file with optimized configuration for readability
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats,           # Complete statistics dictionary
                     f,                     # Destination file
                     indent=2,              # Indentation for readability
                     ensure_ascii=False,    # Preserve special characters
                     default=str)           # Convert non-serializable objects
        
        # Confirm location of saved file
        print(f"💾 Complete report saved to: {report_path}")
        
    def run_complete_analysis(self):
        """
        Execute the complete dataset analysis workflow.
        
        Orchestrates all analysis methods in logical sequence.
        
        Execution Flow:
            1. Analyze directory structure and class distribution
            2. Examine image technical properties via sampling
            3. Generate comprehensive visualizations
            4. Produce intelligent training recommendations
            5. Save complete report in persistent JSON format
        
        Ideal For:
            - Initial evaluation of new datasets
            - Data quality audits
            - Training strategy planning
            - Dataset characteristics documentation
        
        Returns:
            None: Executes all analysis methods and saves results.
        """
        print("🚀 Starting complete dataset analysis...")
        print("="*60)
        
        # Step 1: Analyze directory structure and count images per category
        self.analyze_dataset_structure()
        
        # Step 2: Examine technical properties with representative sample
        self.analyze_image_properties(sample_size=2000)  # Sample of 2000 images
        
        # Step 3: Generate charts and analysis visualizations
        self.create_visualization_report()
        
        # Step 4: Produce recommendations based on findings
        self.generate_recommendations()
        
        # Step 5: Export all results to JSON file
        self.save_analysis_report()
        
        # Confirm successful completion of complete analysis
        print("\n🎉 Analysis completed successfully!")


# Main execution block when script is run directly
if __name__ == "__main__":
    # System path configuration
    # Path to main directory containing YESDOG and NODOG
    dataset_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS"
    
    # Create analyzer instance with specified path
    analyzer = DatasetAnalyzer(dataset_path)
    
    # Execute complete automated analysis
    analyzer.run_complete_analysis()