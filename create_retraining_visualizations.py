# !/usr/bin/env python3
"""
Retraining Analysis Visualization Module.

This module generates comprehensive visualizations to analyze whether model
retraining is necessary and to what extent. It creates diagnostic charts
that help in decision-making about model improvement strategies.

Key Visualizations:
    - Accuracy distribution histogram by class
    - Evaluation criteria vs thresholds comparison
    - Cost-benefit analysis scatter plot
    - Problematic classes bar chart
    - Multi-dimensional radar chart evaluation

Output Files:
    - retraining_analysis_visualization.png: Complete analysis dashboard
    - retraining_recommendation_radar.png: Radar chart evaluation

Usage:
    python create_retraining_visualizations.py

Author: System IA
Date: 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def create_retraining_visualization():
    """
    Generate comprehensive visualizations for retraining necessity analysis.

    This function creates a multi-panel visualization dashboard that analyzes
    whether the current model requires retraining and provides cost-benefit
    analysis for different improvement strategies.

    The function loads evaluation results from a JSON report file and generates:
    1. Accuracy distribution histogram across all classes
    2. Evaluation criteria comparison against thresholds
    3. Cost-benefit scatter plot for improvement options
    4. Horizontal bar chart of most problematic classes
    5. Radar chart for multi-dimensional model evaluation

    Returns:
        None: Saves visualization files to disk.

    Output Files:
        - retraining_analysis_visualization.png: 4-panel analysis dashboard
        - retraining_recommendation_radar.png: Multi-dimensional evaluation

    Raises:
        FileNotFoundError: If complete_class_evaluation_report.json is not found.

    Note:
        Requires matplotlib with polar projection support for radar charts.
    """
    
    # Load data from evaluation report
    workspace_path = Path(r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG")
    eval_file = workspace_path / "complete_class_evaluation_report.json"
    
    if eval_file.exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        print(" Evaluation data not found")
        return
    
    # Configure style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(' RETRAINING NECESSITY ANALYSIS', fontsize=16, fontweight='bold')
    
    # Panel 1: Accuracy distribution histogram by class
    class_details = results.get('class_details', {})
    accuracies = [details['accuracy'] for details in class_details.values()]
    
    ax1.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axvline(np.mean(accuracies), color='red', linestyle='--', 
               label=f'Mean: {np.mean(accuracies):.3f}')
    ax1.axvline(0.7, color='orange', linestyle='--', 
               label='Problematic threshold (0.70)', alpha=0.8)
    ax1.set_xlabel('Accuracy per Class')
    ax1.set_ylabel('Number of Classes')
    ax1.set_title(' Accuracy Distribution by Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Evaluation criteria vs thresholds
    criteria_names = ['Average\nAccuracy', 'Inter-class\nVariability', 
                     'Performance\nGap', 'Problematic\nClasses']
    current_values = [0.865, 0.119, 0.464, 8]
    thresholds = [0.85, 0.15, 0.30, 8]
    
    colors = ['green' if curr <= thresh else 'orange' 
              for curr, thresh in zip(current_values, thresholds)]
    
    bars = ax2.bar(criteria_names, current_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add threshold reference lines
    for i, thresh in enumerate(thresholds):
        ax2.axhline(y=thresh, xmin=i/len(thresholds), xmax=(i+1)/len(thresholds), 
                   color='red', linestyle='--', alpha=0.8)
    
    ax2.set_title(' Evaluation Criteria vs Thresholds')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, current_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 3: Cost-benefit analysis
    options = ['Keep\nCurrent', 'Current\nOptimization', 
               'Targeted\nFine-tuning', 'Complete\nRetraining']
    accuracy_gains = [0.000, 0.020, 0.050, 0.080]
    time_costs = [0, 1, 3, 6]
    
    # Bubble sizes proportional to effort
    sizes = [50, 100, 200, 400]
    colors_cost = ['green', 'lightgreen', 'orange', 'red']
    
    scatter = ax3.scatter(time_costs, accuracy_gains, s=sizes, c=colors_cost, 
                         alpha=0.7, edgecolors='black', linewidths=2)
    
    # Add labels
    for i, option in enumerate(options):
        ax3.annotate(option, (time_costs[i], accuracy_gains[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Implementation Time (weeks)')
    ax3.set_ylabel('Expected Accuracy Gain')
    ax3.set_title(' Cost-Benefit Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Most problematic classes
    problematic_breeds = []
    problematic_accs = []
    
    for breed, details in class_details.items():
        if details['accuracy'] < 0.70:
            problematic_breeds.append(breed.replace('_', ' '))
            problematic_accs.append(details['accuracy'])
    
    # Sort by accuracy
    sorted_data = sorted(zip(problematic_breeds, problematic_accs), 
                        key=lambda x: x[1])
    
    if sorted_data:
        breeds, accs = zip(*sorted_data)
        
        bars4 = ax4.barh(range(len(breeds)), accs, color='lightcoral', 
                        alpha=0.7, edgecolor='darkred')
        ax4.set_yticks(range(len(breeds)))
        ax4.set_yticklabels(breeds, fontsize=9)
        ax4.axvline(x=0.7, color='orange', linestyle='--', 
                   label='Problematic threshold')
        ax4.set_xlabel('Accuracy')
        ax4.set_title(' Most Problematic Classes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars4, accs)):
            ax4.text(acc + 0.01, i, f'{acc:.3f}', 
                    va='center', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No problematic\nclasses', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, fontweight='bold')
        ax4.set_title(' No Problematic Classes')
    
    plt.tight_layout()
    plt.savefig('retraining_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print(" Visualization saved: retraining_analysis_visualization.png")
    
    # Create radar chart for multi-dimensional evaluation
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define evaluation dimensions
    categories = ['General\nAccuracy', 'Low\nVariability', 'No Excessive\nGap', 'Few Problematic\nClasses']
    scores = [0.865/0.95, (0.15-0.119)/0.15, (0.40-0.464)/0.40, (12-8)/12]  # Normalized 0-1
    scores = [max(0, min(1, score)) for score in scores]  # Clamp between 0-1
    
    # Close the radar chart
    scores += scores[:1]
    categories += categories[:1]
    
    angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]
    
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, scores, 'o-', linewidth=2, label='Current Performance')
    ax.fill(angles, scores, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    ax.set_ylim(0, 1)
    ax.set_title(' Multi-dimensional Model Evaluation', 
                size=16, fontweight='bold', pad=20)
    
    # Add "good" threshold reference
    good_threshold = [0.8] * len(categories)
    ax.plot(angles, good_threshold, '--', color='green', alpha=0.7, 
           label='"Good" Threshold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.savefig('retraining_recommendation_radar.png', dpi=300, bbox_inches='tight')
    print(" Recommendation chart saved: retraining_recommendation_radar.png")
    
    print(f"\n VISUALIZATIONS CREATED:")
    print(f"    retraining_analysis_visualization.png - Complete analysis")
    print(f"    retraining_recommendation_radar.png - Multi-dimensional evaluation")


if __name__ == "__main__":
    create_retraining_visualization()