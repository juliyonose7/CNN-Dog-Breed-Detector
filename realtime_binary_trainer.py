# !/usr/bin/env python3
"""
Technical documentation in English.
===============================================
- Train Acc, Val Acc, Learning Rate en tiempo real
Technical documentation in English.
Technical documentation in English.
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

class RealTimeController:
    """Technical documentation in English."""
    
    def __init__(self):
        self.should_stop = False
        self.input_thread = None
        self.epoch_complete = False
    
    def start_monitoring(self):
        """Start monitoreo de entrada"""
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
    
    def _monitor_input(self):
        """Monitor de entrada en hilo separado"""
        while not self.should_stop:
            try:
                if self.epoch_complete:
                    print("\n" + "="*70)
                    print("🛑 ÉPOCA COMPLETADA - ¿Continuar?")
                    print("   ✅ ENTER = Continuar  |  ❌ 'q' + ENTER = Parar")
                    print("="*70)
                    
                    user_input = input(">>> ").strip().lower()
                    if user_input == 'q':
                        print("🛑 Deteniendo entrenamiento...")
                        self.should_stop = True
                    else:
                        print("▶️ Continuando...")
                    
                    self.epoch_complete = False
                
                time.sleep(0.1)
            except (EOFError, KeyboardInterrupt):
                self.should_stop = True
                break
    
    def epoch_finished(self):
        """Technical documentation in English."""
        self.epoch_complete = True
    
    def should_continue(self):
        """Verificar if must continuar"""
        return not self.should_stop

class FastBinaryDataset(Dataset):
    """Dataset binario optimized"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load paths y labels
        self.samples = []
        self.labels = []
        
        # Class 0: NO-PERRO (nodog)
        no_dog_dir = self.data_dir / split / "nodog"
        if no_dog_dir.exists():
            for img_path in no_dog_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append(str(img_path))
                    self.labels.append(0)
        
        # Class 1: PERRO (dog)
        dog_dir = self.data_dir / split / "dog"
        if dog_dir.exists():
            for img_path in dog_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append(str(img_path))
                    self.labels.append(1)
        
        print(f"📊 {split.upper()}: {len(self.samples):,} muestras | NO-PERRO: {sum(1 for l in self.labels if l == 0):,} | PERRO: {sum(1 for l in self.labels if l == 1):,}")
    
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
            # Implementation note.
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

class FastBinaryModel(nn.Module):
    """Model binario with ResNet18"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def calculate_fast_metrics(y_true, y_pred, y_scores):
    """Technical documentation in English."""
    metrics = {}
    
    # Implementation note.
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC for classification binaria
    if len(np.unique(y_true)) == 2:
        metrics['auc'] = roc_auc_score(y_true, y_scores[:, 1])
    
    return metrics

def evaluate_fast(model, dataloader, device):
    """Technical documentation in English."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    return calculate_fast_metrics(all_labels, all_predictions, np.array(all_scores))

def print_header():
    """Technical documentation in English."""
    print("\n" + "="*90)
    print("📊 MÉTRICAS EN TIEMPO REAL")
    print("="*90)
    print(f"{'ÉPOCA':<6} {'TRAIN ACC':<12} {'VAL ACC':<12} {'LEARNING RATE':<15} {'TRAIN LOSS':<12} {'AUC':<8} {'TIEMPO':<8}")
    print("-"*90)

def print_realtime_metrics(epoch, train_acc, val_acc, lr, train_loss, auc, elapsed_time):
    """Technical documentation in English."""
    print(f"{epoch:<6} {train_acc*100:>9.2f}%   {val_acc*100:>9.2f}%   {lr:>12.6f}   {train_loss:>9.4f}   {auc:>6.3f}  {elapsed_time:>6.1f}s")

def main():
    print("🐕 ENTRENADOR BINARIO - MÉTRICAS EN TIEMPO REAL")
    print("🚀 Train Acc | Val Acc | Learning Rate en vivo")
    print("="*80)
    
    # Configuration
    DATA_DIR = "processed_data"
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Dispositivo: {device}")
    print(f"🎯 Épocas: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    
    # Crear directory for models
    os.makedirs("realtime_binary_models", exist_ok=True)
    
    # Transformaciones
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
    print("\n📊 Cargando datasets...")
    train_dataset = FastBinaryDataset(DATA_DIR, 'train', train_transform)
    val_dataset = FastBinaryDataset(DATA_DIR, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Crear model
    print("\n🤖 Creando modelo ResNet18...")
    model = FastBinaryModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Controlador
    controller = RealTimeController()
    controller.start_monitoring()
    
    print("\n⚠️ CONTROL: Después de cada época podrás continuar o parar")
    
    # Implementation note.
    print_header()
    
    # Variables for tracking
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
        
        start_time = time.time()
        
        # training
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []
        train_scores = []
        
        # Implementation note.
        progress_bar = tqdm(train_loader, 
                          desc=f"Época {epoch+1:2d}/{EPOCHS}", 
                          leave=False,
                          ncols=100)
        
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
            
            # Implementation note.
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_scores.extend(scores.detach().cpu().numpy())
            
            # Implementation note.
            if len(train_labels) > 0:
                current_acc = accuracy_score(train_labels, train_predictions)
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{current_acc:.3f}',
                    'LR': f'{current_lr:.5f}'
                })
        
        if not controller.should_continue():
            break
            
        # Implementation note.
        train_metrics = calculate_fast_metrics(train_labels, train_predictions, np.array(train_scores))
        avg_train_loss = train_loss / len(train_loader)
        
        # validation
        val_metrics = evaluate_fast(model, val_loader, device)
        
        # Actualizar scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Tiempo transcurrido
        elapsed_time = time.time() - start_time
        
        # Implementation note.
        print_realtime_metrics(
            epoch + 1,
            train_metrics['accuracy'],
            val_metrics['accuracy'], 
            current_lr,
            avg_train_loss,
            val_metrics.get('auc', 0),
            elapsed_time
        )
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, f"realtime_binary_models/best_model_epoch_{epoch+1}_acc_{val_metrics['accuracy']:.4f}.pth")
            print(f"    💾 ¡Nuevo mejor modelo guardado! (Val Acc: {best_val_acc:.4f})")
        
        # Save log
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': current_lr,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        training_log['epochs'].append(epoch_data)
        
        # Implementation note.
        controller.epoch_finished()
        
        # Esperar hasta that el user decida
        while controller.epoch_complete and controller.should_continue():
            time.sleep(0.1)
    
    # Finalizar training
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_accuracy'] = float(best_val_acc)
    
    log_path = f"realtime_binary_models/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*90)
    print(f"🎉 ENTRENAMIENTO FINALIZADO")
    print(f"🏆 Mejor Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"📄 Log guardado: {log_path}")
    print(f"💾 Modelos en: realtime_binary_models/")
    print("="*90)

if __name__ == "__main__":
    main()