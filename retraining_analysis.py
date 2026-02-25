# !/usr/bin/env python3
"""
Retraining Analysis Module for Dog Breed Classification.
==========================================

This module provides comprehensive analysis of model performance and determines
whether retraining is necessary based on evaluation metrics, bias analysis,
and performance gaps across breed classes.

Author: System IA
Date: 2024
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class RetrainingAnalyzer:
    """Analyzer for evaluating model retraining necessity.
    
    This class loads previous evaluation results, analyzes performance gaps,
    categorizes improvement strategies, and generates action plans based on
    model performance metrics.
    
    Attributes:
        workspace_path: Path to the project workspace.
        bias_analysis: Loaded bias analysis results.
        class_evaluation: Loaded class-level evaluation results.
    """
    
    def __init__(self, workspace_path: str):
        """Initialize the retraining analyzer.
        
        Args:
            workspace_path: Path to the project workspace directory.
        """
        self.workspace_path = Path(workspace_path)
        
        # Load previous evaluation results
        self.load_previous_results()
    
    def load_previous_results(self):
        """Load results from previous model evaluations.
        
        Loads bias analysis and class evaluation reports from JSON files
        to enable comparison and trend analysis.
        """
        self.bias_analysis = {}
        self.class_evaluation = {}
        
        # Load bias analysis report
        bias_file = self.workspace_path / "bias_analysis_report.json"
        if bias_file.exists():
            with open(bias_file, 'r', encoding='utf-8') as f:
                self.bias_analysis = json.load(f)
        
        # Load detailed class evaluation report
        class_file = self.workspace_path / "complete_class_evaluation_report.json"
        if class_file.exists():
            with open(class_file, 'r', encoding='utf-8') as f:
                self.class_evaluation = json.load(f)
    
    def analyze_current_performance_gaps(self):
        """Analyze current performance gaps across breed classes.
        
        Examines class-level accuracy metrics to identify problematic classes,
        calculate performance statistics, and determine if retraining is needed.
        
        Returns:
            dict: Performance analysis containing mean accuracy, standard deviation,
                  performance gap, problematic/excellent classes, and retraining recommendation.
        """
        print("🔍 CURRENT PERFORMANCE GAP ANALYSIS")
        print("="*70)
        
        if not self.class_evaluation:
            print("❌ No class evaluation results found")
            return None
        
        # Identify problematic and excellent classes
        class_details = self.class_evaluation.get('class_details', {})
        problematic_classes = []
        excellent_classes = []
        
        for breed, details in class_details.items():
            accuracy = details.get('accuracy', 0.0)
            if accuracy < 0.7:
                problematic_classes.append((breed, accuracy))
            elif accuracy > 0.95:
                excellent_classes.append((breed, accuracy))
        
        # Calculate overall performance statistics
        accuracies = [details['accuracy'] for details in class_details.values()]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        
        print(f"📊 CURRENT PERFORMANCE STATISTICS:")
        print(f"   Average accuracy: {mean_acc:.3f}")
        print(f"   Standard deviation: {std_acc:.3f}")
        print(f"   Range: {min_acc:.3f} - {max_acc:.3f}")
        print(f"   Problematic classes (<0.70): {len(problematic_classes)}")
        print(f"   Excellent classes (>0.95): {len(excellent_classes)}")
        
        # Calculate performance gap
        performance_gap = max_acc - min_acc
        print(f"   🚨 PERFORMANCE GAP: {performance_gap:.3f}")
        
        # Evaluate retraining necessity
        needs_retraining = self._evaluate_retraining_need(
            mean_acc, std_acc, len(problematic_classes), performance_gap
        )
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'performance_gap': performance_gap,
            'problematic_classes': problematic_classes,
            'excellent_classes': excellent_classes,
            'needs_retraining': needs_retraining
        }
    
    def _evaluate_retraining_need(self, mean_acc, std_acc, problematic_count, gap):
        """Evaluate whether model retraining is necessary.
        
        Args:
            mean_acc: Mean accuracy across all classes.
            std_acc: Standard deviation of accuracy across classes.
            problematic_count: Number of classes with accuracy below threshold.
            gap: Performance gap between best and worst performing classes.
            
        Returns:
            dict: Retraining recommendation with reasons and priority level.
        """
        reasons = []
        priority = "LOW"
        
        # Criteria for retraining
        if std_acc > 0.15:
            reasons.append(f"High variability between classes (std={std_acc:.3f})")
            priority = "MEDIUM"
        
        if gap > 0.4:
            reasons.append(f"Excessive gap between best/worst class ({gap:.3f})")
            priority = "HIGH"
        
        if problematic_count > 8:
            reasons.append(f"Too many problematic classes ({problematic_count})")
            priority = "HIGH"
        
        if mean_acc < 0.85:
            reasons.append(f"Low average accuracy ({mean_acc:.3f})")
            priority = "MEDIUM"
        
        return {
            'recommended': len(reasons) > 0,
            'priority': priority,
            'reasons': reasons
        }
    
    def categorize_improvement_strategies(self):
        """Categorize improvement strategies by retraining requirement.
        
        Organizes improvement strategies into categories based on whether
        they require retraining, fine-tuning, or can be applied as
        post-processing without model changes.
        
        Returns:
            dict: Categorized improvement strategies.
        """
        print("\n🔧 IMPROVEMENT STRATEGY CATEGORIZATION")
        print("="*70)
        
        # Improvements without retraining (already implemented)
        no_retraining = {
            "✅ IMPLEMENTED WITHOUT RETRAINING": [
                "Remove selective model (unified architecture)",
                "Adaptive thresholds per breed", 
                "Detailed metrics per individual class",
                "Optimized temperature calibration",
                "Stratified evaluation per class",
                "Automated bias detection system"
            ]
        }
        
        # Improvements requiring retraining
        requires_retraining = {
            "🔄 REQUIRE COMPLETE RETRAINING": [
                "Geographic dataset diversification (+ Asian/African breeds)",
                "Physical size balancing (+ large breeds)",
                "Data augmentation specific for problematic classes",
                "Improved architecture (e.g., EfficientNet, Vision Transformer)",
                "Transfer learning with more recent models",
                "Multi-task training (detection + classification)"
            ],
            "🎯 REQUIRE TARGETED FINE-TUNING": [
                "Retrain only problematic classes",
                "Learning rate adjustment per class",
                "Weighted loss for imbalanced classes",
                "Focal loss for difficult classes",
                "Class-balanced sampling during training",
                "Mixup/CutMix specific for problematic classes"
            ]
        }
        
        # Post-processing improvements
        post_processing = {
            "⚡ POST-PROCESSING (WITHOUT RETRAINING)": [
                "Ensemble of multiple existing models",
                "Test-time augmentation (TTA)",
                "Advanced probability calibration",
                "Adaptive confidence filters",
                "Voting schemes for ambiguous predictions",
                "Rejection sampling for uncertain predictions"
            ]
        }
        
        all_strategies = {**no_retraining, **requires_retraining, **post_processing}
        
        for category, strategies in all_strategies.items():
            print(f"\n{category}:")
            for i, strategy in enumerate(strategies, 1):
                print(f"   {i}. {strategy}")
        
        return all_strategies
    
    def recommend_action_plan(self, performance_analysis):
        """Generate recommended action plan based on performance analysis.
        
        Args:
            performance_analysis: Dictionary containing performance metrics
                                 and retraining recommendation.
                                 
        Returns:
            dict: Phased action plan with specific steps.
        """
        print(f"\n🎯 ACTION PLAN RECOMMENDATION")
        print("="*70)
        
        if not performance_analysis:
            print("❌ Cannot generate recommendation without performance analysis")
            return None
        
        needs_retraining = performance_analysis['needs_retraining']
        priority = needs_retraining['priority']
        
        print(f"🚦 RETRAINING PRIORITY: {priority}")
        
        if needs_retraining['recommended']:
            print(f"\n✅ RETRAINING RECOMMENDED")
            print(f"📋 Reasons:")
            for reason in needs_retraining['reasons']:
                print(f"   • {reason}")
        else:
            print(f"\n❌ IMMEDIATE RETRAINING NOT REQUIRED")
            print(f"✅ Implemented improvements are sufficient for now")
        
        # Create specific action plan
        action_plan = self._create_specific_action_plan(performance_analysis)
        
        print(f"\n🚀 RECOMMENDED ACTION PLAN:")
        for phase, actions in action_plan.items():
            print(f"\n📋 {phase}:")
            for i, action in enumerate(actions, 1):
                print(f"   {i}. {action}")
        
        return action_plan
    
    def _create_specific_action_plan(self, analysis):
        """Create specific action plan based on analysis results.
        
        Args:
            analysis: Performance analysis dictionary with retraining recommendation.
            
        Returns:
            dict: Phased action plan with specific recommended actions.
        """
        needs_retraining = analysis['needs_retraining']
        problematic_count = len(analysis['problematic_classes'])
        performance_gap = analysis['performance_gap']
        
        if not needs_retraining['recommended']:
            return {
                "PHASE 1 - CURRENT OPTIMIZATION (0-2 weeks)": [
                    "Optimize adaptive thresholds with more validation data",
                    "Implement ensemble of current model with different temperatures",
                    "Apply test-time augmentation to improve predictions",
                    "Monitor performance with detailed per-class metrics"
                ],
                "PHASE 2 - CONTINUOUS EVALUATION": [
                    "Collect feedback from real users",
                    "Analyze specific failure cases", 
                    "Review retraining necessity in 3 months"
                ]
            }
        elif needs_retraining['priority'] == 'MEDIUM':
            return {
                "PHASE 1 - IMPROVEMENTS WITHOUT RETRAINING (1-2 weeks)": [
                    "Implement ensemble of existing models",
                    "Apply advanced calibration techniques",
                    "Test-time augmentation for problematic classes",
                    "Inference hyperparameter optimization"
                ],
                "PHASE 2 - TARGETED FINE-TUNING (2-3 weeks)": [
                    f"Fine-tune only the {problematic_count} most problematic classes",
                    "Apply weighted loss specific for difficult classes",
                    "Intensive data augmentation for problematic classes",
                    "Cross-validation to evaluate improvements"
                ]
            }
        else:  # HIGH priority
            return {
                "PHASE 1 - IMMEDIATE IMPROVEMENTS (1 week)": [
                    "Implement ensemble and TTA to mitigate current issues",
                    "Apply rejection sampling for low confidence predictions",
                    "Document current limitations for users"
                ],
                "PHASE 2 - COMPLETE RETRAINING (3-4 weeks)": [
                    "Collect additional data for problematic classes",
                    "Diversify dataset geographically",
                    "Train with improved architecture (EfficientNet-B4 or ViT)",
                    "Implement advanced balancing techniques"
                ],
                "PHASE 3 - VALIDATION AND DEPLOYMENT (1-2 weeks)": [
                    "Exhaustive evaluation of the new model",
                    "A/B testing against current model",
                    "Gradual deployment and performance monitoring"
                ]
            }
    
    def estimate_improvement_potential(self, action_plan):
        """Estimate improvement potential for each strategy.
        
        Args:
            action_plan: Dictionary with recommended action plan phases.
            
        Returns:
            dict: Improvement estimates for each strategy with accuracy gain,
                  time investment, cost, and success probability.
        """
        print(f"\n📈 IMPROVEMENT POTENTIAL ESTIMATION")
        print("="*70)
        
        current_performance = self.class_evaluation.get('overall_accuracy', 0.868)
        
        improvement_estimates = {
            "Optimización actual (sin reentrenamiento)": {
                "accuracy_gain": 0.02,  # +2%
                "time_investment": "1-2 semanas",
                "cost": "Bajo",
                "probability_success": 0.9
            },
            "Fine-tuning dirigido": {
                "accuracy_gain": 0.05,  # +5%
                "time_investment": "2-3 semanas", 
                "cost": "Medio",
                "probability_success": 0.7
            },
            "Reentrenamiento completo": {
                "accuracy_gain": 0.08,  # +8%
                "time_investment": "4-6 semanas",
                "cost": "Alto", 
                "probability_success": 0.6
            }
        }
        
        print(f"📊 CURRENT PERFORMANCE: {current_performance:.3f}")
        print(f"\n🎯 IMPROVEMENT ESTIMATES:")
        
        for strategy, estimates in improvement_estimates.items():
            projected_acc = current_performance + estimates['accuracy_gain']
            expected_gain = estimates['accuracy_gain'] * estimates['probability_success']
            
            print(f"\n📋 {strategy}:")
            print(f"   🎯 Estimated gain: +{estimates['accuracy_gain']:.3f} ({estimates['accuracy_gain']*100:.1f}%)")
            print(f"   📈 Projected accuracy: {projected_acc:.3f}")
            print(f"   📊 Expected gain: +{expected_gain:.3f} ({expected_gain*100:.1f}%)")
            print(f"   ⏰ Time: {estimates['time_investment']}")
            print(f"   💰 Cost: {estimates['cost']}")
            print(f"   📈 Success probability: {estimates['probability_success']*100:.0f}%")
        
        return improvement_estimates
    
    def create_decision_matrix(self, performance_analysis, improvement_estimates):
        """Create decision matrix visualization for strategy selection.
        
        Args:
            performance_analysis: Performance analysis results dictionary.
            improvement_estimates: Dictionary with improvement estimates per strategy.
            
        Returns:
            dict: Decision matrix with recommended strategy and rationale.
        """
        print(f"\n📊 DECISION MATRIX")
        print("="*70)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Expected gain vs time
        strategies = list(improvement_estimates.keys())
        gains = [est['accuracy_gain'] * est['probability_success'] for est in improvement_estimates.values()]
        times = [1.5, 2.5, 5.0]  # Average weeks
        costs = ['Low', 'Medium', 'High']
        colors = ['green', 'orange', 'red']
        
        scatter = ax1.scatter(times, gains, s=200, c=colors, alpha=0.7)
        ax1.set_xlabel('Implementation Time (weeks)')
        ax1.set_ylabel('Expected Accuracy Gain')
        ax1.set_title('Expected Gain vs Implementation Time')
        ax1.grid(True, alpha=0.3)
        
        # Add labels
        for i, (strategy, gain, time) in enumerate(zip(strategies, gains, times)):
            ax1.annotate(strategy.split('(')[0], (time, gain), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Plot 2: Accuracy distribution per class
        if self.class_evaluation and 'class_details' in self.class_evaluation:
            class_details = self.class_evaluation['class_details']
            accuracies = [details['accuracy'] for details in class_details.values()]
            
            ax2.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='navy')
            ax2.axvline(np.mean(accuracies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(accuracies):.3f}')
            ax2.axvline(0.7, color='orange', linestyle='--', 
                       label='Problematic threshold')
            ax2.set_xlabel('Accuracy per Class')
            ax2.set_ylabel('Number of Classes')
            ax2.set_title('Accuracy Distribution per Class')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig('retraining_decision_matrix.png', dpi=300, bbox_inches='tight')
        print("   ✅ Decision matrix saved: retraining_decision_matrix.png")
        
        # Generate final recommendation
        needs_retraining = performance_analysis['needs_retraining']
        
        if needs_retraining['priority'] == 'LOW':
            recommendation = "Current optimization"
            rationale = "Implemented improvements are sufficient. Optimize without retraining."
        elif needs_retraining['priority'] == 'MEDIUM':
            recommendation = "Targeted fine-tuning" 
            rationale = "Optimal balance between expected improvement and required effort."
        else:
            recommendation = "Complete retraining"
            rationale = "Current issues require fundamental intervention."
        
        print(f"\n🎯 FINAL RECOMMENDATION: {recommendation}")
        print(f"💡 Rationale: {rationale}")
        
        return {
            'recommended_strategy': recommendation,
            'rationale': rationale,
            'visualization_path': 'retraining_decision_matrix.png'
        }
    
    def run_complete_analysis(self):
        """Run complete retraining necessity analysis.
        
        Executes all analysis steps including performance gap analysis,
        strategy categorization, action plan recommendation, improvement
        estimation, and decision matrix generation.
        
        Returns:
            dict: Complete analysis report with all results and recommendations.
        """
        print("🔬" * 70)
        print("🔬 COMPLETE RETRAINING NECESSITY ANALYSIS")
        print("🔬" * 70)
        
        # 1. Analyze current performance
        performance_analysis = self.analyze_current_performance_gaps()
        
        # 2. Categorize strategies
        strategies = self.categorize_improvement_strategies()
        
        # 3. Recommend action plan
        action_plan = self.recommend_action_plan(performance_analysis)
        
        # 4. Estimate improvement potential
        improvement_estimates = self.estimate_improvement_potential(action_plan)
        
        # 5. Create decision matrix
        decision_matrix = self.create_decision_matrix(performance_analysis, improvement_estimates)
        
        # 6. Save complete report
        complete_report = {
            'timestamp': np.datetime64('now').item().isoformat(),
            'performance_analysis': performance_analysis,
            'improvement_strategies': strategies,
            'action_plan': action_plan,
            'improvement_estimates': improvement_estimates,
            'decision_matrix': decision_matrix
        }
        
        with open('retraining_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(complete_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ COMPLETE ANALYSIS FINISHED")
        print(f"   📊 Report saved: retraining_analysis_report.json")
        print(f"   📈 Visualization: retraining_decision_matrix.png")
        
        return complete_report

def main():
    """Main entry point for retraining analysis."""
    workspace_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
    
    analyzer = RetrainingAnalyzer(workspace_path)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()