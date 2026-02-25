# !/usr/bin/env python3
"""
Retraining of the model main with dataset balanced
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class BalancedBreedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        
        # Get all the breeds and create mapping
        breeds = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
        
        for idx, breed in enumerate(breeds):
            self.class_to_idx[breed] = idx
            breed_path = os.path.join(data_dir, breed)
            
            for img_file in os.listdir(breed_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(breed_path, img_file), idx))
        
        print(f"📊 Dataset balanceado cargado:")
        print(f"   Total imágenes: {len(self.samples)}")
        print(f"   Total clases: {len(self.class_to_idx)}")
        print(f"   Promedio por clase: {len(self.samples) / len(self.class_to_idx):.1f}")
        
        # Verify balance
        class_counts = {}
        for _, class_idx in self.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        counts = list(class_counts.values())
        print(f"   Min/Max por clase: {min(counts)}/{max(counts)}")
        print(f"   Desviación estándar: {np.std(counts):.2f}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            # Implementation note.
            return self.__getitem__((idx + 1) % len(self.samples))

class ImprovedBreedClassifier(nn.Module):
    """Model mejorado with better architecture"""
    def __init__(self, num_classes=50):
        super().__init__()
        # Use ResNet50 for best performance
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        
        # Congelar the primeras layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        # Reemplazar clasificador final
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def train_balanced_model():
    """Entrenar model with dataset balanced"""
    
    print("🚀 ENTRENAMIENTO CON DATASET BALANCEADO")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Dispositivo: {device}")
    
    num_classes = 50
    batch_size = 16
    num_epochs = 25
    learning_rate = 0.001
    
    # Implementation note.
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
    
    # Dataset complete
    full_dataset = BalancedBreedDataset(
        'breed_processed_data/train',
        transform=None  # Implementation note.
    )
    
    # Split train/validation
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Implementation note.
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    print(f"📊 División de datos:")
    print(f"   Entrenamiento: {len(train_dataset):,} imágenes")
    print(f"   Validación: {len(val_dataset):,} imágenes")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Model
    model = ImprovedBreedClassifier(num_classes).to(device)
    
    # Loss and optimizer with class weights balanceados
    criterion = nn.CrossEntropyLoss()  # No necesitamos weights with dataset balanced
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=3, 
        factor=0.5
    )
    
    # Variables for tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print(f"\n🎯 Iniciando entrenamiento ({num_epochs} épocas)...")
    
    for epoch in range(num_epochs):
        # === training ===
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_preds += target.size(0)
            correct_preds += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Época {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_preds / total_preds
        
        # === validation ===
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        # Implementation note.
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Época {epoch+1}/{num_epochs}:")
        print(f"  Train: Loss {epoch_train_loss:.4f}, Acc {epoch_train_acc:.2f}%")
        print(f"  Val:   Loss {epoch_val_loss:.4f}, Acc {epoch_val_acc:.2f}%")
        
        # Scheduler step
        scheduler.step(epoch_val_acc)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            
            # Create directory for models balanceados
            os.makedirs('balanced_models', exist_ok=True)
            
            # save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accuracy': epoch_train_acc,
                'val_accuracy': epoch_val_acc,
                'class_to_idx': full_dataset.class_to_idx,
                'model_architecture': 'ResNet50',
                'dataset_balanced': True,
                'images_per_class': 161
            }, f'balanced_models/best_balanced_breed_model_epoch_{epoch+1}_acc_{epoch_val_acc:.4f}.pth')
            
            print(f"✅ Nuevo mejor modelo guardado: {epoch_val_acc:.2f}%")
    
    print(f"\n🏆 Mejor accuracy de validación: {best_val_acc:.2f}%")
    
    # Implementation note.
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Pérdida durante entrenamiento')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.title('Precisión durante entrenamiento')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('balanced_training_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Implementation note.
    metrics = {
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accuracies[-1],
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'num_epochs': num_epochs,
        'dataset_info': {
            'balanced': True,
            'images_per_class': 161,
            'total_classes': num_classes,
            'total_images': len(full_dataset)
        }
    }
    
    with open('balanced_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Resultados guardados:")
    print(f"   📊 Métricas: balanced_training_metrics.json")
    print(f"   📈 Gráficos: balanced_training_metrics.png")
    print(f"   🤖 Modelo: balanced_models/best_balanced_breed_model_epoch_*")
    
    return best_val_acc, full_dataset.class_to_idx

if __name__ == "__main__":
    # Implementation note.
    if not os.path.exists('balancing_final_report.json'):
        print("❌ Primero ejecuta balance_dataset.py")
        exit(1)
    
    # Entrenar model
    best_acc, class_mapping = train_balanced_model()
    
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
    print(f"🏆 Mejor accuracy: {best_acc:.2f}%")
    print(f"⚖️ Dataset perfectamente balanceado usado")
    print(f"🚀 Listo para integrar el nuevo modelo")