#!/usr/bin/env python3
"""
🐕 ENTRENADOR BINARIO AVANZADO
===============================
- Métricas completas (Accuracy, Precision, Recall, F1-Score, AUC)
- Control manual después de cada época
- Visualización detallada del progreso
- Guardado automático del mejor modelo
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

class EnhancedController:
    """Controlador mejorado para parar entrenamiento después de cada época"""
    
    def __init__(self):
        self.should_stop = False
        self.input_thread = None
        self.epoch_complete = False
    
    def start_monitoring(self):
        """Iniciar monitoreo de entrada"""
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
    
    def _monitor_input(self):
        """Monitor de entrada en hilo separado"""
        while not self.should_stop:
            try:
                if self.epoch_complete:
                    print("\n" + "="*80)
                    print("🛑 ÉPOCA COMPLETADA - ¿Continuar entrenamiento?")
                    print("   ✅ Presiona ENTER para continuar")
                    print("   ❌ Escribe 'q' + ENTER para parar")
                    print("="*80)
                    
                    user_input = input(">>> ").strip().lower()
                    if user_input == 'q':
                        print("🛑 Deteniendo entrenamiento por solicitud del usuario...")
                        self.should_stop = True
                    else:
                        print("▶️ Continuando con la siguiente época...")
                    
                    self.epoch_complete = False
                
                time.sleep(0.1)
            except (EOFError, KeyboardInterrupt):
                self.should_stop = True
                break
    
    def epoch_finished(self):
        """Marcar época como completada"""
        self.epoch_complete = True
    
    def should_continue(self):
        """Verificar si debe continuar"""
        return not self.should_stop

class EnhancedBinaryDataset(Dataset):
    """Dataset binario optimizado para carga rápida"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Cargar paths y labels
        self.samples = []
        self.labels = []
        
        # Clase 0: NO-PERRO (nodog)
        no_dog_dir = self.data_dir / split / "nodog"
        if no_dog_dir.exists():
            for img_path in no_dog_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append(str(img_path))
                    self.labels.append(0)
        
        # Clase 1: PERRO (dog)
        dog_dir = self.data_dir / split / "dog"
        if dog_dir.exists():
            for img_path in dog_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append(str(img_path))
                    self.labels.append(1)
        
        print(f"📊 Dataset {split}: {len(self.samples)} muestras")
        print(f"   ❌ NO-PERRO: {sum(1 for l in self.labels if l == 0):,}")
        print(f"   ✅ PERRO: {sum(1 for l in self.labels if l == 1):,}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Retornar imagen negra en caso de error
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

class EnhancedBinaryModel(nn.Module):
    """Modelo binario con ResNet18 y métricas avanzadas"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def calculate_metrics(y_true, y_pred, y_scores):
    """Calcular métricas completas"""
    metrics = {}
    
    # Métricas básicas
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC (solo para clasificación binaria)
    if len(np.unique(y_true)) == 2:
        metrics['auc'] = roc_auc_score(y_true, y_scores[:, 1])
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Métricas adicionales
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

def print_metrics(metrics, split_name=""):
    """Imprimir métricas de forma bonita"""
    print(f"\n📊 MÉTRICAS {split_name.upper()}")
    print("="*60)
    print(f"🎯 Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"🎯 Precision:  {metrics['precision']:.4f}")
    print(f"🎯 Recall:     {metrics['recall']:.4f}")
    print(f"🎯 F1-Score:   {metrics['f1']:.4f}")
    
    if 'auc' in metrics:
        print(f"📈 AUC:        {metrics['auc']:.4f}")
    
    if 'specificity' in metrics and 'sensitivity' in metrics:
        print(f"🔍 Specificity: {metrics['specificity']:.4f}")
        print(f"🔍 Sensitivity: {metrics['sensitivity']:.4f}")
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\n📋 MATRIZ DE CONFUSIÓN:")
        print(f"    Pred:  [NO-PERRO] [PERRO]")
        print(f"Real NO-PERRO:  {cm[0,0]:6d}   {cm[0,1]:6d}")
        print(f"Real PERRO:     {cm[1,0]:6d}   {cm[1,1]:6d}")

def evaluate_model(model, dataloader, device):
    """Evaluar modelo con métricas completas"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluando", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    return calculate_metrics(all_labels, all_predictions, np.array(all_scores))

def save_training_log(log_data, log_path):
    """Guardar log de entrenamiento"""
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)

def main():
    print("🐕 ENTRENADOR BINARIO AVANZADO")
    print("🚀 Con métricas completas y control por época")
    print("="*80)
    
    # Configuración
    DATA_DIR = "processed_data"
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Dispositivo: {device}")
    
    # Crear directorio para modelos
    os.makedirs("enhanced_binary_models", exist_ok=True)
    
    # Transformaciones optimizadas
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Crear datasets
    print("📊 Creando datasets...")
    train_dataset = EnhancedBinaryDataset(DATA_DIR, 'train', train_transform)
    val_dataset = EnhancedBinaryDataset(DATA_DIR, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Crear modelo
    print("🤖 Creando modelo ResNet18...")
    model = EnhancedBinaryModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Controlador mejorado
    controller = EnhancedController()
    controller.start_monitoring()
    
    print("\n🚀 ENTRENAMIENTO BINARIO AVANZADO")
    print("="*80)
    print(f"🎯 Épocas: {EPOCHS}")
    print(f"🔄 Batch Size: {BATCH_SIZE}")
    print(f"📚 Learning Rate: {LEARNING_RATE}")
    print(f"💻 Dispositivo: {device}")
    print("⚠️ El sistema te preguntará después de cada época si continuar")
    
    # Variables para tracking
    best_val_acc = 0
    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'device': str(device)
        },
        'epochs': []
    }
    
    for epoch in range(EPOCHS):
        if not controller.should_continue():
            print("🛑 Entrenamiento detenido por el usuario")
            break
            
        print(f"\n📅 ÉPOCA {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        # ENTRENAMIENTO
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []
        train_scores = []
        
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}")
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            if not controller.should_continue():
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Recopilar predicciones para métricas
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_scores.extend(scores.detach().cpu().numpy())
            
            # Actualizar barra de progreso
            current_acc = accuracy_score(train_labels[-len(labels):], 
                                       train_predictions[-len(labels):])
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        if not controller.should_continue():
            break
            
        # Calcular métricas de entrenamiento
        train_metrics = calculate_metrics(train_labels, train_predictions, np.array(train_scores))
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDACIÓN
        print("\n🔍 Evaluando en validación...")
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Actualizar scheduler
        scheduler.step()
        
        # Imprimir resultados
        print(f"\n🏃 RESULTADOS ÉPOCA {epoch + 1}")
        print("="*60)
        print(f"📉 Train Loss: {avg_train_loss:.4f}")
        print_metrics(train_metrics, "TRAIN")
        print_metrics(val_metrics, "VALIDACIÓN")
        
        # Guardar mejor modelo
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, f"enhanced_binary_models/best_model_epoch_{epoch+1}_acc_{val_metrics['accuracy']:.4f}.pth")
            print(f"💾 Mejor modelo guardado! (Acc: {best_val_acc:.4f})")
        
        # Guardar log
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': scheduler.get_last_lr()[0],
            'timestamp': datetime.now().isoformat()
        }
        training_log['epochs'].append(epoch_data)
        
        # Marcar época completada y esperar decisión del usuario
        controller.epoch_finished()
        
        # Esperar hasta que el usuario decida
        while controller.epoch_complete and controller.should_continue():
            time.sleep(0.1)
    
    # Finalizar entrenamiento
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_accuracy'] = float(best_val_acc)
    
    log_path = f"enhanced_binary_models/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_training_log(training_log, log_path)
    
    print(f"\n🎉 ENTRENAMIENTO FINALIZADO")
    print("="*60)
    print(f"🏆 Mejor accuracy de validación: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"📄 Log guardado en: {log_path}")
    print("✅ Todos los modelos guardados en: enhanced_binary_models/")

if __name__ == "__main__":
    main()