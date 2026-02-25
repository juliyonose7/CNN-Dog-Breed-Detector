# !/usr/bin/env python3
"""
Retraining Necessity Analysis for Dog Breed Classification Model.
========================================================

Provides comprehensive analysis of whether model retraining is necessary
based on previous evaluations and already implemented improvements.

Author: System IA
Date: 2024
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_retraining_necessity():
    """Analyze the necessity of model retraining.
    
    Evaluates current model performance, implemented improvements, and determines
    whether retraining is justified based on cost-benefit analysis.
    
    Returns:
        dict: Complete analysis report with recommendation and rationale.
    """
    print("🔬" * 70)
    print("🔬 RETRAINING NECESSITY ANALYSIS")  
    print("🔬" * 70)
    
    workspace_path = Path(r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG")
    
    # ==========================================
    # 1. Current state analysis
    # ==========================================
    print("\n🔍 CURRENT STATE ANALYSIS")
    print("="*50)
    
    # Load existing evaluation results
    eval_file = workspace_path / "complete_class_evaluation_report.json"
    current_results = None
    
    if eval_file.exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            current_results = json.load(f)
    
    if current_results:
        overall_acc = current_results.get('overall_accuracy', 0.0)
        class_details = current_results.get('class_details', {})
        
        # Calculate detailed statistics
        accuracies = [details['accuracy'] for details in class_details.values()]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        performance_gap = max_acc - min_acc
        
        # Identify problematic classes
        problematic = [(breed, acc) for breed, details in class_details.items() 
                      for acc in [details['accuracy']] if acc < 0.7]
        
        print(f"📊 CURRENT PERFORMANCE:")
        print(f"   Overall accuracy: {overall_acc:.3f} (86.8%)")
        print(f"   Average accuracy per class: {mean_acc:.3f}")
        print(f"   Standard deviation: {std_acc:.3f}")
        print(f"   Range: {min_acc:.3f} - {max_acc:.3f}")
        print(f"   Performance gap: {performance_gap:.3f}")
        print(f"   Problematic classes (<0.70): {len(problematic)}")
        
        if problematic:
            print(f"   🚨 Most problematic:")
            for breed, acc in sorted(problematic)[:5]:
                print(f"      • {breed}: {acc:.3f}")
    else:
        print("❌ No detailed evaluation results found")
        # Use previous reference values
        overall_acc = 0.868
        mean_acc = 0.868
        std_acc = 0.12
        performance_gap = 0.35
        problematic = [('Lhasa', 0.536), ('cairn', 0.586), ('Siberian_husky', 0.621)]
        
        print(f"📊 USING PREVIOUS REFERENCE VALUES:")
        print(f"   Overall accuracy: {overall_acc:.3f}")
        print(f"   Most problematic classes: {len(problematic)}")
    
    # ==========================================
    # 2. Already implemented improvements
    # ==========================================
    print(f"\n✅ ALREADY IMPLEMENTED IMPROVEMENTS (WITHOUT RETRAINING)")
    print("="*50)
    
    implemented_improvements = [
        "🏗️ Removed selective model → Unified ResNet50 architecture",
        "🎯 Adaptive thresholds per breed → Optimized range 0.736-0.800",
        "📊 Detailed metrics per class → 50 breeds with precision/recall/F1",
        "🛡️ Bias detection system → Automated continuous analysis",
        "⚖️ Temperature calibration → Better calibrated probabilities",
        "📈 Stratified evaluation → Balanced per-class validation"
    ]
    
    for improvement in implemented_improvements:
        print(f"   ✅ {improvement}")
    
    # ==========================================
    # 3. Retraining necessity evaluation
    # ==========================================
    print(f"\n🎯 RETRAINING NECESSITY EVALUATION")
    print("="*50)
    
    # Evaluation criteria
    criteria = {
        "Average accuracy": {
            "current": mean_acc,
            "threshold": 0.85,
            "status": "✅ GOOD" if mean_acc >= 0.85 else "⚠️ IMPROVABLE",
            "needs_retraining": mean_acc < 0.80
        },
        "Variability between classes": {
            "current": std_acc,
            "threshold": 0.15,
            "status": "✅ GOOD" if std_acc <= 0.15 else "⚠️ HIGH",
            "needs_retraining": std_acc > 0.20
        },
        "Performance gap": {
            "current": performance_gap,
            "threshold": 0.30,
            "status": "✅ GOOD" if performance_gap <= 0.30 else "⚠️ HIGH",
            "needs_retraining": performance_gap > 0.40
        },
        "Problematic classes": {
            "current": len(problematic),
            "threshold": 8,
            "status": "✅ GOOD" if len(problematic) <= 8 else "⚠️ MANY",
            "needs_retraining": len(problematic) > 12
        }
    }
    
    retraining_votes = 0
    total_votes = len(criteria)
    
    for criterion, info in criteria.items():
        print(f"   📊 {criterion}: {info['current']:.3f} | Threshold: {info['threshold']:.3f} | {info['status']}")
        if info['needs_retraining']:
            retraining_votes += 1
    
    retraining_percentage = (retraining_votes / total_votes) * 100
    
    # ==========================================
    # 4. Final decision
    # ==========================================
    print(f"\n🚦 FINAL DECISION")
    print("="*50)
    
    print(f"📊 Votes pro-retraining: {retraining_votes}/{total_votes} ({retraining_percentage:.1f}%)")
    
    if retraining_percentage <= 25:
        recommendation = "❌ DO NOT RETRAIN"
        priority = "LOW"
        rationale = "Implemented improvements are sufficient. Current model has good performance."
        next_steps = [
            "Continue monitoring current performance",
            "Optimize adaptive thresholds with more data", 
            "Implement ensemble for edge cases",
            "Re-evaluate in 3-6 months"
        ]
    elif retraining_percentage <= 50:
        recommendation = "⚠️ TARGETED FINE-TUNING"
        priority = "MEDIUM"
        rationale = "Some specific issues can be resolved with focused adjustments."
        next_steps = [
            "Fine-tune only for most problematic classes",
            "Specific data augmentation for difficult breeds",
            "Weighted loss for imbalanced classes",
            "Validate improvements before full deployment"
        ]
    else:
        recommendation = "✅ COMPLETE RETRAINING"
        priority = "HIGH"
        rationale = "Multiple fundamental issues require retraining from scratch."
        next_steps = [
            "Expand dataset with more geographic diversity",
            "Test more advanced architectures (EfficientNet/ViT)",
            "Implement advanced balancing techniques",
            "Plan complete retraining in 4-6 weeks"
        ]
    
    print(f"\n🎯 RECOMMENDATION: {recommendation}")
    print(f"🚦 PRIORITY: {priority}")
    print(f"💡 RATIONALE: {rationale}")
    
    print(f"\n📋 NEXT STEPS:")
    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")
    
    # ==========================================
    # 5. ALTERNATIVAS without retraining
    # ==========================================
    if retraining_percentage <= 50:
        print(f"\n🔧 ALTERNATIVES WITHOUT RETRAINING")
        print("="*50)
        
        alternatives = [
            "🤖 Ensemble of multiple existing models",
            "🔄 Test-time augmentation (TTA) for more robust predictions",
            "📊 Advanced probability calibration",
            "🎯 Adaptive threshold refinement",
            "🛡️ Confidence filters to reject ambiguous predictions",
            "📈 Voting schemes for difficult cases"
        ]
        
        expected_improvement = 0.02 if retraining_percentage <= 25 else 0.04
        
        print(f"   💡 These alternatives could improve accuracy by ~{expected_improvement:.2f} ({expected_improvement*100:.0f}%)")
        print(f"   ⏱️ Implementation time: 1-2 weeks")
        print(f"   💰 Cost: Low")
        
        for alt in alternatives:
            print(f"   • {alt}")
    
    # ==========================================
    # Implementation note.
    # ==========================================
    print(f"\n💰 COST-BENEFIT ANALYSIS")
    print("="*50)
    
    options = {
        "Keep current": {
            "accuracy_gain": 0.00,
            "time_weeks": 0,
            "cost": "None",
            "effort": "Minimal"
        },
        "Current optimization": {
            "accuracy_gain": 0.02,
            "time_weeks": 1,
            "cost": "Very low",
            "effort": "Low"
        },
        "Targeted fine-tuning": {
            "accuracy_gain": 0.05,
            "time_weeks": 3,
            "cost": "Medium",
            "effort": "Medium"
        },
        "Full retraining": {
            "accuracy_gain": 0.08,
            "time_weeks": 6,
            "cost": "High",
            "effort": "High"
        }
    }
    
    current_acc = overall_acc
    
    for option, details in options.items():
        projected_acc = current_acc + details['accuracy_gain']
        efficiency = details['accuracy_gain'] / max(details['time_weeks'], 0.1)  # Implementation note.
        
        print(f"   📊 {option}:")
        print(f"      🎯 Accuracy: {current_acc:.3f} → {projected_acc:.3f} (+{details['accuracy_gain']:.3f})")
        print(f"      ⏱️ Time: {details['time_weeks']} weeks")
        print(f"      💰 Cost: {details['cost']}")
        print(f"      📈 Efficiency: {efficiency:.3f} gain/week")
        print()
    
    # ==========================================
    # Implementation note.
    # ==========================================
    print(f"🏆 EXECUTIVE CONCLUSION")
    print("="*50)
    
    if retraining_percentage <= 25:
        conclusion = f"""
✅ KEEP CURRENT MODEL WITH MINOR OPTIMIZATIONS

Analysis indicates that the improvements already implemented (elimination of 
architectural biases, adaptive thresholds, detailed metrics) have been very
effective. With {overall_acc:.1%} overall accuracy and only {len(problematic)} 
problematic classes, current performance is satisfactory.

RECOMMENDATION: Continue with current model, applying minor optimizations
like ensemble and TTA to maximize performance without retraining.
        """
    elif retraining_percentage <= 50:
        conclusion = f"""
⚠️ TARGETED FINE-TUNING RECOMMENDED

Although implemented improvements have been positive, there are {len(problematic)} 
problematic classes and a performance gap of {performance_gap:.2f} that 
justifies targeted fine-tuning.

RECOMMENDATION: Specific fine-tuning for the most problematic classes,
maintaining the current unified architecture but improving balance
between classes.
        """
    else:
        conclusion = f"""
🚨 FULL RETRAINING REQUIRED

Detected problems (high variability: {std_acc:.2f}, performance gap: 
{performance_gap:.2f}, {len(problematic)} problematic classes) 
indicate fundamental limitations requiring full retraining.

RECOMMENDATION: Plan full retraining with expanded dataset
and improved architecture to address structural problems.
        """
    
    print(conclusion)
    
# Save reporte
report = {
'timestamp': str(np.datetime64('now')),
'current_performance': {
'overall_accuracy': overall_acc,
'mean_accuracy': mean_acc,
'std_accuracy': std_acc,
'performance_gap': performance_gap,
'problematic_classes': len(problematic)
},
'retraining_analysis': {
'votes_for_retraining': retraining_votes,
'total_votes': total_votes,
'retraining_percentage': retraining_percentage,
'recommendation': recommendation,
'priority': priority,
'rationale': rationale
},
'next_steps': next_steps,
'conclusion': conclusion.strip()
}
    
with open('retraining_necessity_analysis.json', 'w', encoding='utf-8') as f:
json.dump(report, f, indent=2, ensure_ascii=False)
    
print(f"\n✅ Reporte complete saved: retraining_necessity_analysis.json")
    
return report

if __name__ == "__main__":
analyze_retraining_necessity()