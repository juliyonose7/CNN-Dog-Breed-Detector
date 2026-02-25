# !/usr/bin/env python3
"""
Technical documentation in English.
=========================================================
Technical documentation in English.
- 50 breeds with Train Acc, Val Acc, Learning Rate
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

class AutonomousController:
    """Technical documentation in English."""
    
    def __init__(self, check_interval=3):
        self.should_stop = False
        self.input_thread = None
        self.check_complete = False
        self.check_interval = check_interval
        self.epochs_completed = 0
    
    def start_monitoring(self):
        """Start monitoreo de entrada"""
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
    
    def _monitor_input(self):
        """Monitor de entrada en hilo separado"""
        while not self.should_stop:
            try:
                if self.check_complete:
                    print("\n" + "="*80)
                    print(f"🛑 COMPLETADAS {self.epochs_completed} ÉPOCAS - ¿Continuar?")
                    print("   ✅ ENTER = Continuar 3 épocas más")
                    print("   ❌ 'q' + ENTER = Parar entrenamiento")
                    print("   📊 's' + ENTER = Mostrar resumen detallado")
                    print("="*80)
                    
                    user_input = input(">>> ").strip().lower()
                    if user_input == 'q':
                        print("🛑 Deteniendo entrenamiento...")
                        self.should_stop = True
                    elif user_input == 's':
                        print("📊 Continuando con resumen detallado...")
                    else:
                        print(f"▶️ Continuando {self.check_interval} épocas más...")
                    
                    self.check_complete = False
                
                time.sleep(0.1)
            except (EOFError, KeyboardInterrupt):
                self.should_stop = True
                break
    
    def epoch_finished(self):
        """Technical documentation in English."""
        self.epochs_completed += 1
        if self.epochs_completed % self.check_interval == 0:
            self.check_complete = True
    
    def should_continue(self):
        """Verificar if must continuar"""
        return not self.should_stop
    
    def should_pause(self):
        """Verificar if must pausar for consulta"""
        return self.check_complete

class BreedDataset(Dataset):
    """Dataset optimized for 50 breeds"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load paths, labels y mapping de classes
        self.samples = []
        self.labels = []
        self.class_names = []
        
        split_dir = self.data_dir / split
        if split_dir.exists():
            class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            
            for class_idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                self.class_names.append(class_name)
                
                # Load images de this class
                class_samples = 0
                for img_path in class_dir.rglob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)
                        class_samples += 1
        
        self.num_classes = len(self.class_names)
        print(f"🏷️ {split.upper()}: {len(self.samples):,} muestras | {self.num_classes} razas")
    
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

class BreedModel(nn.Module):
    """Model for classification de breeds with ResNet34"""
    
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = models.resnet34(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def calculate_breed_metrics(y_true, y_pred):
    """Technical documentation in English."""
    metrics = {}
    
    # Implementation note.
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return metrics

def calculate_topk_accuracy(outputs, targets, k=3):
    """Calcular Top-K accuracy"""
    _, pred_topk = outputs.topk(k, 1, True, True)
    pred_topk = pred_topk.t()
    correct = pred_topk.eq(targets.view(1, -1).expand_as(pred_topk))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / targets.size(0)).item()

def evaluate_breed_model(model, dataloader, device):
    """Technical documentation in English."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_top3_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Top-3 accuracy
            top3_acc = calculate_topk_accuracy(outputs, labels, k=3)
            all_top3_correct += (top3_acc * labels.size(0)) / 100
            total_samples += labels.size(0)
    
    metrics = calculate_breed_metrics(all_labels, all_predictions)
    metrics['top3_acc'] = all_top3_correct / total_samples
    
    return metrics

def print_breed_header():
    """Technical documentation in English."""
    print("\n" + "="*100)
    print("🐕 MÉTRICAS DE RAZAS - ENTRENAMIENTO AUTÓNOMO (CADA 3 ÉPOCAS)")
    print("="*100)
    print(f"{'ÉPOCA':<6} {'TRAIN ACC':<12} {'VAL ACC':<12} {'TOP-3 ACC':<12} {'LR':<12} {'LOSS':<10} {'F1':<8} {'TIEMPO':<8}")
    print("-"*100)

def print_breed_metrics(epoch, train_acc, val_acc, top3_acc, lr, train_loss, f1, elapsed_time):
    """Technical documentation in English."""
    print(f"{epoch:<6} {train_acc*100:>9.2f}%   {val_acc*100:>9.2f}%   {top3_acc*100:>9.2f}%   {lr:>9.6f}  {train_loss:>7.4f}  {f1:>6.3f}  {elapsed_time:>6.1f}s")

def print_progress_summary(epochs_completed, best_val_acc, best_top3_acc, avg_time_per_epoch):
    """Technical documentation in English."""
    print("\n" + "🔥" * 50)
    print(f"📊 RESUMEN DE PROGRESO - {epochs_completed} ÉPOCAS COMPLETADAS")
    print("🔥" * 50)
    print(f"🏆 Mejor Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"🥉 Mejor Top-3 Accuracy: {best_top3_acc:.4f} ({best_top3_acc*100:.2f}%)")
    print(f"⏱️ Tiempo promedio por época: {avg_time_per_epoch:.1f} segundos")
    print(f"🚀 Progreso esperado: {'Excelente' if best_val_acc > 0.6 else 'Muy bueno' if best_val_acc > 0.4 else 'Normal'}")
    print("🔥" * 50)

def main():
    print("🐕 ENTRENADOR DE RAZAS AUTÓNOMO")
    print("🚀 50 Razas | Entrenamiento cada 3 épocas | Métricas en tiempo real")
    print("="*80)
    
    # Configuration optimizada for 50 classes
    DATA_DIR = "breed_processed_data"
    BATCH_SIZE = 12
    EPOCHS = 25
    LEARNING_RATE = 0.0005
    CHECK_INTERVAL = 3  # Implementation note.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Dispositivo: {device}")
    print(f"🎯 Total Épocas: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"⏸️ Control cada: {CHECK_INTERVAL} épocas")
    
    # Crear directory for models
    os.makedirs("autonomous_breed_models", exist_ok=True)
    
    # Transformaciones optimizadas for breeds
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Crear datasets
    print("\n📊 Cargando datasets de razas...")
    train_dataset = BreedDataset(DATA_DIR, 'train', train_transform)
    val_dataset = BreedDataset(DATA_DIR, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Crear model
    print(f"\n🤖 Creando modelo ResNet34 para {train_dataset.num_classes} razas...")
    model = BreedModel(num_classes=train_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    # Implementation note.
    controller = AutonomousController(check_interval=CHECK_INTERVAL)
    controller.start_monitoring()
    
    print(f"\n⚠️ CONTROL AUTÓNOMO: El sistema entrenará {CHECK_INTERVAL} épocas y preguntará si continuar")
    print("💡 Top-3 Accuracy: % de veces que la raza correcta está en las 3 predicciones más probables")
    
    # Implementation note.
    print_breed_header()
    
    # Variables for tracking
    best_val_acc = 0
    best_top3_acc = 0
    epoch_times = []
    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'check_interval': CHECK_INTERVAL,
            'num_classes': train_dataset.num_classes,
            'device': str(device)
        },
        'class_names': train_dataset.class_names,
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
        
        # Implementation note.
        progress_bar = tqdm(train_loader, 
                          desc=f"Época {epoch+1:2d}/{EPOCHS} [50 razas] - Autónomo", 
                          leave=False,
                          ncols=120)
        
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
            
            # Recopilar predictions
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Implementation note.
            if len(train_labels) > 0:
                current_acc = accuracy_score(train_labels, train_predictions)
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{current_acc:.3f}',
                    'LR': f'{current_lr:.6f}'
                })
        
        if not controller.should_continue():
            break
            
        # Implementation note.
        train_metrics = calculate_breed_metrics(train_labels, train_predictions)
        avg_train_loss = train_loss / len(train_loader)
        
        # validation
        val_metrics = evaluate_breed_model(model, val_loader, device)
        
        # Actualizar scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Tiempo transcurrido
        elapsed_time = time.time() - start_time
        epoch_times.append(elapsed_time)
        
        # Implementation note.
        print_breed_metrics(
            epoch + 1,
            train_metrics['accuracy'],
            val_metrics['accuracy'],
            val_metrics['top3_acc'],
            current_lr,
            avg_train_loss,
            val_metrics['f1'],
            elapsed_time
        )
        
        # Save best model
        improved = False
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            improved = True
            
        if val_metrics['top3_acc'] > best_top3_acc:
            best_top3_acc = val_metrics['top3_acc']
            improved = True
            
        if improved:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_top3_acc': best_top3_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'class_names': train_dataset.class_names
            }, f"autonomous_breed_models/best_breed_model_epoch_{epoch+1}_acc_{val_metrics['accuracy']:.4f}.pth")
            print(f"    💾 Mejor modelo guardado! (Val: {best_val_acc:.4f}, Top-3: {best_top3_acc:.4f})")
        
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
        
        # Implementation note.
        if (epoch + 1) % CHECK_INTERVAL == 0:
            avg_time = sum(epoch_times[-CHECK_INTERVAL:]) / CHECK_INTERVAL
            print_progress_summary(epoch + 1, best_val_acc, best_top3_acc, avg_time)
        
        # Implementation note.
        while controller.should_pause() and controller.should_continue():
            time.sleep(0.1)
    
    # Finalizar training
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_accuracy'] = float(best_val_acc)
    training_log['best_top3_accuracy'] = float(best_top3_acc)
    
    log_path = f"autonomous_breed_models/autonomous_breed_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*100)
    print(f"🎉 ENTRENAMIENTO AUTÓNOMO DE RAZAS FINALIZADO")
    print(f"🏆 Mejor Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"🥉 Mejor Top-3 Accuracy: {best_top3_acc:.4f} ({best_top3_acc*100:.2f}%)")
    print(f"⏱️ Tiempo promedio por época: {sum(epoch_times)/len(epoch_times):.1f}s")
    print(f"📄 Log guardado: {log_path}")
    print(f"💾 Modelos en: autonomous_breed_models/")
    print("="*100)

if __name__ == "__main__":
    main()