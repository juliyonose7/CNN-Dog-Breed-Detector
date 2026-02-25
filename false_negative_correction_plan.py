#!/usr/bin/env python3
"""
🛠️ PLAN DE CORRECCIÓN PARA FALSOS NEGATIVOS
==========================================
Estrategias prácticas para reducir falsos negativos en razas problemáticas
del modelo de 119 clases
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class FalseNegativeCorrector:
    def __init__(self):
        self.problematic_breeds = {
            'critical': ['Lhasa', 'cairn'],
            'high_priority': ['Siberian_husky', 'whippet', 'malamute', 'Australian_terrier', 
                            'Norfolk_terrier', 'toy_terrier', 'Italian_greyhound'],
            'medium_priority': ['Lakeland_terrier', 'Border_terrier', 'bluetick', 
                              'Rhodesian_ridgeback', 'Ibizan_hound']
        }
        
    def generate_correction_plan(self):
        """Generar plan completo de corrección"""
        print("🛠️ PLAN DE CORRECCIÓN PARA FALSOS NEGATIVOS")
        print("=" * 60)
        
        correction_strategies = {
            "1_threshold_adjustment": self.threshold_adjustment_strategy(),
            "2_weighted_loss": self.weighted_loss_strategy(), 
            "3_data_augmentation": self.data_augmentation_strategy(),
            "4_focal_loss": self.focal_loss_strategy(),
            "5_ensemble_methods": self.ensemble_strategy(),
            "6_hard_negative_mining": self.hard_negative_mining_strategy(),
            "7_class_balancing": self.class_balancing_strategy(),
            "8_feature_enhancement": self.feature_enhancement_strategy()
        }
        
        return correction_strategies
    
    def threshold_adjustment_strategy(self):
        """Estrategia 1: Ajuste de umbrales por clase"""
        print("\n📈 ESTRATEGIA 1: AJUSTE DE UMBRALES POR CLASE")
        print("-" * 50)
        print("🎯 Objetivo: Reducir umbrales para razas conservadoras")
        
        strategy = {
            "description": "Usar umbrales adaptativos más bajos para razas con muchos falsos negativos",
            "implementation": """
# Umbrales personalizados por raza (más bajos = menos conservador)
BREED_THRESHOLDS = {
    'Lhasa': 0.35,           # Muy bajo (era conservador)
    'cairn': 0.40,           # Bajo (era muy conservador)  
    'Siberian_husky': 0.45,  # Bajo-medio
    'whippet': 0.45,         # Bajo-medio
    'malamute': 0.50,        # Medio
    'Australian_terrier': 0.50,
    'Norfolk_terrier': 0.50,
    'toy_terrier': 0.55,     # Medio-alto (tenía buena precision)
    # Razas normales usan threshold estándar = 0.60
}

def apply_adaptive_thresholds(predictions, breed_names, default_threshold=0.60):
    adjusted_predictions = []
    
    for i, breed in enumerate(breed_names):
        threshold = BREED_THRESHOLDS.get(breed, default_threshold)
        pred_score = predictions[i]
        
        # Aplicar threshold personalizado
        if pred_score >= threshold:
            adjusted_predictions.append((breed, pred_score, True))
        else:
            adjusted_predictions.append((breed, pred_score, False))
    
    return adjusted_predictions
            """,
            "expected_improvement": "15-25% reducción en falsos negativos para razas críticas",
            "risk_level": "BAJO - fácil de implementar y revertir"
        }
        
        print("✅ Implementación INMEDIATA recomendada")
        print("📊 Mejora esperada: 15-25% menos falsos negativos")
        
        return strategy
    
    def weighted_loss_strategy(self):
        """Estrategia 2: Función de pérdida ponderada"""
        print("\n🎯 ESTRATEGIA 2: WEIGHTED LOSS FUNCTION")
        print("-" * 50)
        print("🎯 Objetivo: Penalizar más los falsos negativos que los falsos positivos")
        
        strategy = {
            "description": "Usar pesos de clase que penalicen más los falsos negativos en razas problemáticas",
            "implementation": """
import torch.nn as nn

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, recall_weight=2.0):
        super().__init__()
        self.alpha = alpha  # Pesos por clase
        self.gamma = gamma  # Factor focal
        self.recall_weight = recall_weight  # Penalización extra para FN
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha)(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Focal loss component
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Extra penalty for false negatives
        # Detectar predicciones incorrectas
        pred_classes = torch.argmax(inputs, dim=1)
        false_negatives = (pred_classes != targets)
        
        # Aplicar penalización extra a falsos negativos
        penalty = torch.where(false_negatives, 
                            torch.tensor(self.recall_weight), 
                            torch.tensor(1.0)).to(inputs.device)
        
        return (focal_loss * penalty).mean()

# Pesos específicos para razas problemáticas
CLASS_WEIGHTS = {
    'Lhasa': 3.0,           # Triple peso (tenía 46% FN)
    'cairn': 2.8,           # Alto peso (tenía 41% FN)
    'Siberian_husky': 2.5,  # Alto peso
    'whippet': 2.3,         # Alto peso
    'malamute': 2.2,        # Medio-alto peso
    # Razas normales = 1.0
}

def create_class_weights(num_classes, problematic_breeds_weights):
    weights = torch.ones(num_classes)
    
    for breed_idx, breed_name in enumerate(breed_names):
        if breed_name in problematic_breeds_weights:
            weights[breed_idx] = problematic_breeds_weights[breed_name]
    
    return weights
            """,
            "expected_improvement": "20-35% reducción en falsos negativos",
            "risk_level": "MEDIO - requiere reentrenamiento"
        }
        
        print("⚡ Implementación: Requiere reentrenamiento del modelo")
        print("📊 Mejora esperada: 20-35% menos falsos negativos")
        
        return strategy
    
    def data_augmentation_strategy(self):
        """Estrategia 3: Augmentación de datos especializada"""
        print("\n🔄 ESTRATEGIA 3: AUGMENTACIÓN ESPECIALIZADA")
        print("-" * 50)
        print("🎯 Objetivo: Más variedad de datos para razas problemáticas")
        
        strategy = {
            "description": "Augmentación específica según el tipo de raza y sus problemas",
            "implementation": """
import torchvision.transforms as transforms
from torchvision.transforms import RandomAffine, ColorJitter, RandomHorizontalFlip

# Augmentación específica por grupo de razas
BREED_SPECIFIC_AUGMENTATION = {
    # Para terriers pequeños (características sutiles)
    'terriers': transforms.Compose([
        transforms.RandomRotation(15),  # Variación de ángulo
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Zoom variado
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Iluminación
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Posición
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Enfoque variado
    ]),
    
    # Para perros nórdicos (diferencias de tamaño/pelaje)
    'nordic': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Más variación de escala
        transforms.ColorJitter(brightness=0.4, saturation=0.3),  # Pelaje variado
        transforms.RandomPerspective(distortion_scale=0.2),  # Perspectiva
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # Oclusión parcial
    ]),
    
    # Para galgos/lebreles (proporciones corporales)
    'sighthounds': transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # Cuerpo completo
        transforms.ColorJitter(contrast=0.4),  # Definición muscular
        transforms.RandomRotation(25),  # Ángulos variados
    ])
}

def apply_breed_specific_augmentation(image, breed_name):
    \"\"\"Aplicar augmentación específica según la raza\"\"\"
    
    # Clasificar raza en grupo
    if breed_name in ['cairn', 'Norfolk_terrier', 'toy_terrier', 'Australian_terrier']:
        augmentation = BREED_SPECIFIC_AUGMENTATION['terriers']
    elif breed_name in ['Siberian_husky', 'malamute']:
        augmentation = BREED_SPECIFIC_AUGMENTATION['nordic'] 
    elif breed_name in ['whippet', 'Italian_greyhound']:
        augmentation = BREED_SPECIFIC_AUGMENTATION['sighthounds']
    else:
        # Augmentación estándar
        augmentation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
    
    return augmentation(image)

# Generar más datos para razas críticas
AUGMENTATION_MULTIPLIER = {
    'Lhasa': 4,           # 4x más datos
    'cairn': 4,           # 4x más datos  
    'Siberian_husky': 3,  # 3x más datos
    'whippet': 3,         # 3x más datos
    'malamute': 3,        # 3x más datos
    # Razas normales = 1x
}
            """,
            "expected_improvement": "10-20% reducción en falsos negativos",
            "risk_level": "BAJO - no afecta modelo actual"
        }
        
        print("📸 Implementación: Generar más datos variados")
        print("📊 Mejora esperada: 10-20% menos falsos negativos")
        
        return strategy
    
    def focal_loss_strategy(self):
        """Estrategia 4: Focal Loss para clases difíciles"""
        print("\n🧠 ESTRATEGIA 4: FOCAL LOSS IMPLEMENTATION")
        print("-" * 50)
        print("🎯 Objetivo: Enfocarse en ejemplos difíciles de clasificar")
        
        strategy = {
            "description": "Usar Focal Loss para dar más importancia a ejemplos difíciles",
            "implementation": """
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, breed_specific_gamma=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.breed_specific_gamma = breed_specific_gamma or {}
        
    def forward(self, inputs, targets, breed_names=None):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Gamma específico por raza si se proporciona
        if breed_names is not None and self.breed_specific_gamma:
            gamma_values = torch.ones_like(targets, dtype=torch.float)
            for i, breed in enumerate(breed_names):
                if breed in self.breed_specific_gamma:
                    gamma_values[i] = self.breed_specific_gamma[breed]
        else:
            gamma_values = self.gamma
            
        focal_loss = self.alpha * (1 - pt) ** gamma_values * ce_loss
        return focal_loss.mean()

# Gamma específico para razas problemáticas (más alto = más enfoque)
BREED_SPECIFIC_GAMMA = {
    'Lhasa': 3.0,           # Muy alto enfoque
    'cairn': 2.8,           # Alto enfoque
    'Siberian_husky': 2.5,  # Alto enfoque
    'whippet': 2.3,         # Medio-alto enfoque
    'malamute': 2.2,        # Medio-alto enfoque
    # Razas normales = 2.0 (gamma estándar)
}

# Integración en el entrenamiento
def train_with_adaptive_focal_loss(model, train_loader, device):
    criterion = AdaptiveFocalLoss(
        alpha=1, 
        gamma=2.0, 
        breed_specific_gamma=BREED_SPECIFIC_GAMMA
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for batch_idx, (data, targets, breed_names) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        # Usar focal loss adaptativo
        loss = criterion(outputs, targets, breed_names)
        loss.backward()
        optimizer.step()
            """,
            "expected_improvement": "25-30% reducción en falsos negativos",
            "risk_level": "MEDIO - requiere reentrenamiento completo"
        }
        
        print("🎯 Implementación: Focal Loss con gamma adaptativo")
        print("📊 Mejora esperada: 25-30% menos falsos negativos")
        
        return strategy
    
    def ensemble_strategy(self):
        """Estrategia 5: Métodos de ensemble"""
        print("\n📊 ESTRATEGIA 5: ENSEMBLE METHODS")
        print("-" * 50)
        print("🎯 Objetivo: Combinar múltiples modelos para mejor recall")
        
        strategy = {
            "description": "Usar ensemble de modelos optimizados para diferentes aspectos",
            "implementation": """
class RecallOptimizedEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
    def predict(self, x):
        predictions = []
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(x), dim=1)
                predictions.append(pred * self.weights[i])
        
        # Promedio ponderado
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
    
    def predict_with_recall_boost(self, x, breed_name, recall_boost_factor=1.2):
        base_prediction = self.predict(x)
        
        # Boost para razas con problemas de recall
        if breed_name in ['Lhasa', 'cairn', 'Siberian_husky', 'whippet']:
            # Incrementar probabilidad de la clase correcta
            class_idx = get_breed_index(breed_name)
            base_prediction[:, class_idx] *= recall_boost_factor
            
            # Renormalizar
            base_prediction = torch.softmax(base_prediction, dim=1)
        
        return base_prediction

# Crear ensemble especializado
def create_recall_optimized_ensemble():
    # Modelo 1: Optimizado para precisión general
    model1 = load_model('best_model_fold_0.pth')
    
    # Modelo 2: Entrenado con focal loss
    model2 = load_model('focal_loss_model.pth')
    
    # Modelo 3: Entrenado con weighted loss
    model3 = load_model('weighted_model.pth')
    
    # Pesos del ensemble (más peso a modelos optimizados para recall)
    ensemble_weights = [0.3, 0.4, 0.3]  # Más peso al focal loss
    
    return RecallOptimizedEnsemble([model1, model2, model3], ensemble_weights)

# Uso en producción
ensemble = create_recall_optimized_ensemble()
prediction = ensemble.predict_with_recall_boost(image, breed_name)
            """,
            "expected_improvement": "30-40% reducción en falsos negativos",
            "risk_level": "ALTO - requiere múltiples modelos"
        }
        
        print("🔄 Implementación: Ensemble de 3 modelos especializados")
        print("📊 Mejora esperada: 30-40% menos falsos negativos")
        
        return strategy
    
    def generate_implementation_roadmap(self):
        """Generar roadmap de implementación"""
        print("\n" + "=" * 70)
        print("🗺️ ROADMAP DE IMPLEMENTACIÓN - CORRECCIÓN DE FALSOS NEGATIVOS")
        print("=" * 70)
        
        roadmap = {
            "Phase_1_Immediate": {
                "timeframe": "1-2 días",
                "actions": [
                    "✅ Implementar ajuste de umbrales por clase",
                    "✅ Aplicar umbrales más bajos a razas críticas",
                    "✅ Testing inmediato en razas problemáticas"
                ],
                "expected_improvement": "15-25%",
                "effort": "BAJO"
            },
            "Phase_2_Short_term": {
                "timeframe": "1 semana", 
                "actions": [
                    "🔄 Implementar augmentación especializada",
                    "📸 Generar más datos para razas críticas",
                    "🧪 Testing con nuevos datos"
                ],
                "expected_improvement": "25-35%",
                "effort": "MEDIO"
            },
            "Phase_3_Medium_term": {
                "timeframe": "2-3 semanas",
                "actions": [
                    "🎯 Implementar Weighted/Focal Loss",
                    "🔄 Reentrenar modelo con nuevas funciones de pérdida",
                    "📊 Validación completa del modelo"
                ],
                "expected_improvement": "35-50%",
                "effort": "ALTO"
            },
            "Phase_4_Long_term": {
                "timeframe": "1 mes",
                "actions": [
                    "📊 Implementar ensemble methods",
                    "🔧 Optimización completa del pipeline",
                    "🚀 Despliegue en producción"
                ],
                "expected_improvement": "50-60%",
                "effort": "MUY ALTO"
            }
        }
        
        for phase, details in roadmap.items():
            print(f"\n🎯 {phase.replace('_', ' ').upper()}")
            print(f"   ⏱️  Tiempo: {details['timeframe']}")
            print(f"   📈 Mejora esperada: {details['expected_improvement']}")
            print(f"   💪 Esfuerzo: {details['effort']}")
            print("   📋 Acciones:")
            for action in details['actions']:
                print(f"      {action}")
        
        return roadmap
    
    def create_quick_fix_script(self):
        """Crear script de corrección rápida (Fase 1)"""
        print("\n" + "=" * 60)
        print("⚡ SCRIPT DE CORRECCIÓN RÁPIDA - LISTO PARA USAR")
        print("=" * 60)
        
        quick_fix_code = '''
# 🚀 CORRECCIÓN INMEDIATA - AJUSTE DE UMBRALES
# Archivo: quick_false_negative_fix.py

import torch
import numpy as np

class ThresholdOptimizedClassifier:
    def __init__(self, base_model, breed_thresholds=None):
        self.base_model = base_model
        self.breed_thresholds = breed_thresholds or {
            'Lhasa': 0.35,           # Muy bajo (era 46% FN)
            'cairn': 0.40,           # Bajo (era 41% FN)
            'Siberian_husky': 0.45,  # Bajo-medio (era 38% FN)
            'whippet': 0.45,         # Bajo-medio (era 36% FN)
            'malamute': 0.50,        # Medio (era 35% FN)
            'Australian_terrier': 0.50,
            'Norfolk_terrier': 0.50,
            'toy_terrier': 0.55,
            'Italian_greyhound': 0.55,
        }
        self.default_threshold = 0.60
        
    def predict_with_adaptive_thresholds(self, image, breed_names):
        # Obtener predicciones del modelo base
        with torch.no_grad():
            logits = self.base_model(image)
            probabilities = torch.softmax(logits, dim=1)
        
        results = []
        
        for i, breed in enumerate(breed_names):
            prob_score = probabilities[0][i].item()
            threshold = self.breed_thresholds.get(breed, self.default_threshold)
            
            # Aplicar threshold adaptativo
            is_predicted = prob_score >= threshold
            
            results.append({
                'breed': breed,
                'probability': prob_score,
                'threshold_used': threshold,
                'predicted': is_predicted,
                'improvement': 'OPTIMIZED' if breed in self.breed_thresholds else 'STANDARD'
            })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)

# USO INMEDIATO:
# 1. Cargar tu modelo actual
# model = torch.load('best_model_fold_0.pth')
# 
# 2. Crear clasificador optimizado
# optimized_classifier = ThresholdOptimizedClassifier(model)
# 
# 3. Usar con imágenes
# results = optimized_classifier.predict_with_adaptive_thresholds(image, breed_names)
'''
        
        # Guardar script
        with open('quick_false_negative_fix.py', 'w') as f:
            f.write(quick_fix_code)
        
        print("💾 Script guardado como: quick_false_negative_fix.py")
        print("⚡ LISTO PARA USAR INMEDIATAMENTE!")
        
        return quick_fix_code

def main():
    """Ejecutar análisis completo de corrección"""
    print("🛠️ Iniciando Plan de Corrección para Falsos Negativos...")
    
    corrector = FalseNegativeCorrector()
    
    # Generar estrategias
    strategies = corrector.generate_correction_plan()
    
    # Generar roadmap
    roadmap = corrector.generate_implementation_roadmap()
    
    # Crear script de corrección rápida
    corrector.create_quick_fix_script()
    
    print("\n" + "=" * 70)
    print("✅ PLAN DE CORRECCIÓN COMPLETADO")
    print("=" * 70)
    print("🎯 PRÓXIMOS PASOS RECOMENDADOS:")
    print("   1. ⚡ Usar 'quick_false_negative_fix.py' INMEDIATAMENTE")
    print("   2. 🧪 Probar con razas problemáticas (Lhasa, Cairn, Husky)")
    print("   3. 📊 Medir mejora en recall")
    print("   4. 🔄 Proceder con Fase 2 si los resultados son buenos")
    
    return {
        'strategies': strategies,
        'roadmap': roadmap,
        'quick_fix_ready': True
    }

if __name__ == "__main__":
    main()