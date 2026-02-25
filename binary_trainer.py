"""
🐕 ENTRENADOR BINARIO CANINO - AMD 7800X3D optimized
Technical documentation in English.
"""

import os
import sys
import time
import json
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import warnings
warnings.filterwarnings('ignore')

# Implementation note.
if sys.platform == "win32":
    import msvcrt
else:
    import select

# Control of parada manual
class TrainingController:
    def __init__(self):
        self.should_stop = False
        self.input_thread = None
        self.monitoring = False
    
    def start_monitoring(self):
        """Inicia the monitoreo of input of the user"""
        self.monitoring = True
        self.should_stop = False
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
        print("🔄 CONTROL DE ENTRENAMIENTO ACTIVADO")
        print("   Presiona 'q' + Enter para parar el entrenamiento de forma segura")
        print("   Presiona 's' + Enter para mostrar estadísticas")
        print("=" * 70)
    
    def _monitor_input(self):
        """Monitorea the input of the user en a hilo separado"""
        while self.monitoring:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        if key == 'q':
                            print("\n🛑 PARADA SOLICITADA - Terminando época actual de forma segura...")
                            self.should_stop = True
                            break
                        elif key == 's':
                            print(f"\n📊 STATUS: Entrenamiento en progreso... (Presiona 'q' para parar)")
                else:
                    # For sistemas Unix/Linux
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.readline().strip().lower()
                        if key == 'q':
                            print("\n🛑 PARADA SOLICITADA - Terminando época actual de forma segura...")
                            self.should_stop = True
                            break
                        elif key == 's':
                            print(f"\n📊 STATUS: Entrenamiento en progreso... (Presiona 'q' para parar)")
                
                time.sleep(0.1)
            except:
                time.sleep(0.1)
                continue
    
    def stop_monitoring(self):
        """Detiene the monitoreo"""
        self.monitoring = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)

# ===================================================================
# configuration AMD 7800X3D
# ===================================================================
def optimize_for_7800x3d():
    """Technical documentation in English."""
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '16'
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['OPENBLAS_NUM_THREADS'] = '16'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '16'
    
    # Implementation note.
    torch.set_num_threads(16)
    torch.set_num_interop_threads(4)
    
    print("🚀 Variables de entorno 7800X3D configuradas")
    print(f"💻 CPU threads: {torch.get_num_threads()}")

# ===================================================================
# DATASET BINARIO
# ===================================================================
class BinaryDogDataset(Dataset):
    """Dataset for classification binaria dog/no-dog"""
    
    def __init__(self, data_path, transform=None, max_samples_per_class=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        self.classes = ['no_dog', 'dog']  # 0: no-dog, 1: dog
        
        self._load_samples(max_samples_per_class)
        
    def _load_samples(self, max_samples_per_class):
        """Load samples balanceadas"""
        print("🔄 Cargando dataset binario...")
        
        # Load images NO-dog
        nodog_path = self.data_path / "NODOG"
        if nodog_path.exists():
            nodog_count = 0
            
            # Files individuales
            for img_file in nodog_path.glob("*.jpg"):
                if max_samples_per_class and nodog_count >= max_samples_per_class:
                    break
                self.samples.append((str(img_file), 0))
                nodog_count += 1
                
            # Subdirectories
            for subdir in nodog_path.iterdir():
                if subdir.is_dir():
                    for img_file in subdir.glob("*.jpg"):
                        if max_samples_per_class and nodog_count >= max_samples_per_class:
                            break
                        self.samples.append((str(img_file), 0))
                        nodog_count += 1
                        
            print(f"   ❌ NO-PERRO: {nodog_count:,} imágenes")
        
        # Load images dog
        yesdog_path = self.data_path / "YESDOG"
        if yesdog_path.exists():
            dog_count = 0
            
            for breed_dir in yesdog_path.iterdir():
                if breed_dir.is_dir():
                    # Search tanto .JPEG como .jpg
                    for img_file in list(breed_dir.glob("*.JPEG")) + list(breed_dir.glob("*.jpg")):
                        if max_samples_per_class and dog_count >= max_samples_per_class:
                            break
                        self.samples.append((str(img_file), 1))
                        dog_count += 1
                        
            print(f"   ✅ PERRO: {dog_count:,} imágenes")
        
        # Balancear dataset
        if max_samples_per_class:
            self._balance_dataset(max_samples_per_class)
        
        print(f"🎯 Total samples: {len(self.samples):,}")
        
    def _balance_dataset(self, target_size):
        """Technical documentation in English."""
        # Separar for class
        no_dog_samples = [s for s in self.samples if s[1] == 0]
        dog_samples = [s for s in self.samples if s[1] == 1]
        
        # Tomar target_size of cada class
        no_dog_balanced = no_dog_samples[:target_size]
        dog_balanced = dog_samples[:target_size]
        
        self.samples = no_dog_balanced + dog_balanced
        
        print(f"⚖️  Dataset balanceado: {len(no_dog_balanced)} no-perro + {len(dog_balanced)} perro")
        
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
            print(f"Error loading {img_path}: {e}")
            # Retornar image en negro como fallback
            fallback = torch.zeros((3, 224, 224))
            return fallback, label

# ===================================================================
# model BINARIO
# ===================================================================
class BinaryDogClassifier(nn.Module):
    """Model binario for classification dog/no-dog"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Implementation note.
        self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Get dimensiones of features
        num_features = self.backbone.classifier[1].in_features
        
        # Reemplazar clasificador
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 2)  # Binario: 0=no-dog, 1=dog
        )
        
    def forward(self, x):
        return self.backbone(x)

# ===================================================================
# ENTRENADOR
# ===================================================================
class BinaryTrainer:
    """Entrenador optimized for classification binaria"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.controller = TrainingController()  # Control of parada manual
        
        # Configurar optimizador and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Criterio with weights balanceados
        self.criterion = nn.CrossEntropyLoss()
        
    def setup_scheduler(self, train_loader, epochs):
        """Configura the scheduler OneCycleLR"""
        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.003,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        print(f"📈 OneCycleLR configurado: {total_steps:,} pasos totales")
        
    def train_epoch(self, train_loader, epoch):
        """Technical documentation in English."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Época {epoch}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Implementation note.
            if self.controller.should_stop:
                print(f"\n🛑 Parada solicitada durante entrenamiento - terminando época...")
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Implementation note.
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{current_lr:.2e}'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, current_lr
    
    def validate(self, val_loader, epoch):
        """Valid the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Validación {epoch}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_preds, all_targets
    
    def train_model(self, train_loader, val_loader, epochs, save_path='./binary_models', patience=5):
        """Training complete"""
        print(f"🚀 INICIANDO ENTRENAMIENTO BINARIO")
        print("=" * 60)
        print(f"🎯 Épocas: {epochs}")
        print(f"🤖 Modelo: EfficientNet-B1")
        print(f"💻 Dispositivo: {self.device}")
        print()
        
        # Start control of parada manual
        self.controller.start_monitoring()
        
        # Configurar scheduler
        self.setup_scheduler(train_loader, epochs)
        
        # Create directory of saved
        Path(save_path).mkdir(exist_ok=True)
        
        best_val_acc = 0
        patience_counter = 0
        
        try:
            for epoch in range(1, epochs + 1):
                # Implementation note.
                if self.controller.should_stop:
                    print(f"\n🛑 ENTRENAMIENTO DETENIDO POR USUARIO EN ÉPOCA {epoch}")
                    break
                
                print(f"📅 ÉPOCA {epoch}/{epochs}")
                print("-" * 40)
                
                # Entrenar
                train_loss, train_acc, current_lr = self.train_epoch(train_loader, epoch)
                
                # Validar
                val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, epoch)
                
                # Implementation note.
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Imprimir resultados
                print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"🔄 Learning Rate: {current_lr:.2e}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    # save model
                    model_path = Path(save_path) / 'best_binary_model.pth'
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'accuracy': val_acc,
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, model_path)
                
                    print(f"✅ Mejor modelo guardado: {val_acc:.2f}% accuracy")
                else:
                    patience_counter += 1
                    print(f"⏳ Paciencia: {patience_counter}/{patience}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"🛑 Early stopping en época {epoch}")
                    break
                
                print()
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Entrenamiento interrumpido manualmente")
        finally:
            # Stop control of parada
            self.controller.stop_monitoring()
        
        # Generar reporte final
        self._generate_report(save_path, best_val_acc, val_preds, val_targets)
        
        return {
            'best_accuracy': best_val_acc,
            'final_epoch': epoch,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }
    
    def _generate_report(self, save_path, best_accuracy, preds, targets):
        """Genera reporte of training"""
        print("📊 GENERANDO REPORTE FINAL...")
        
        # Implementation note.
        plt.figure(figsize=(15, 5))
        
        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(self.val_losses, label='Val Loss', color='red', alpha=0.7)
        plt.title('📉 Pérdida durante Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train Acc', color='green', alpha=0.7)
        plt.plot(self.val_accuracies, label='Val Acc', color='orange', alpha=0.7)
        plt.title('📈 Precisión durante Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Implementation note.
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(targets, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No-Perro', 'Perro'],
                   yticklabels=['No-Perro', 'Perro'])
        plt.title('🎯 Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        
        plt.tight_layout()
        plt.savefig(Path(save_path) / 'binary_training_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reporte of classification
        class_names = ['No-Perro', 'Perro']
        report = classification_report(targets, preds, target_names=class_names)
        
        with open(Path(save_path) / 'binary_classification_report.txt', 'w') as f:
            f.write("🐕 REPORTE DE CLASIFICACIÓN BINARIA\n")
            f.write("=" * 50 + "\n")
            f.write(f"🎯 Mejor Accuracy: {best_accuracy:.2f}%\n")
            f.write(f"💻 Optimizado para: AMD 7800X3D\n")
            f.write(f"🤖 Arquitectura: EfficientNet-B1\n\n")
            f.write(report)
        
        print(f"✅ Reporte guardado en {save_path}")

# ===================================================================
# configuration of data
# ===================================================================
def get_transforms():
    """Gets the transformaciones of data"""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders(data_path, train_transform, val_transform, batch_size=16, num_workers=14):
    """Creates the dataloaders optimizados"""
    print("🔄 CREANDO DATALOADERS BINARIOS...")
    
    # Implementation note.
    train_dataset = BinaryDogDataset(
        data_path=data_path,
        transform=train_transform,
        max_samples_per_class=15000  # 15k for class = 30k total
    )
    
    val_dataset = BinaryDogDataset(
        data_path=data_path,
        transform=val_transform,
        max_samples_per_class=3000   # 3k for class = 6k total
    )
    
    # Create DataLoaders optimizados for 7800X3D
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"✅ DataLoaders creados:")
    print(f"   🏋️  Train: {len(train_dataset):,} samples")
    print(f"   ✅ Val: {len(val_dataset):,} samples")
    print(f"   ⚙️  Batch size: {batch_size}")
    print(f"   👷 Workers: {num_workers}")
    
    return train_loader, val_loader

# ===================================================================
# function main
# ===================================================================
def main():
    """Function main of training"""
    print("🐕 ENTRENADOR BINARIO CANINO")
    print("🚀 Optimizado para AMD 7800X3D")
    print("=" * 80)
    
    # Configurar environment
    optimize_for_7800x3d()
    
    # Configuration
    DATA_PATH = "./DATASETS"
    BATCH_SIZE = 16  # Implementation note.
    EPOCHS = 20
    NUM_WORKERS = 14  # 7800X3D tiene 16 threads, dejar 2 for system
    
    # Verify data
    if not Path(DATA_PATH).exists():
        print(f"❌ Directorio de datos no encontrado: {DATA_PATH}")
        return
    
    # Create model
    print("🤖 Creando modelo...")
    model = BinaryDogClassifier(pretrained=True)
    
    # Configurar device
    device = torch.device('cpu')
    print(f"💻 Usando dispositivo: {device}")
    
    # Create trainer
    trainer = BinaryTrainer(model, device)
    
    # Preparar data
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = create_dataloaders(
        DATA_PATH, train_transform, val_transform, BATCH_SIZE, NUM_WORKERS
    )
    
    # Entrenar
    start_time = time.time()
    results = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_path='./binary_models',
        patience=5
    )
    
    training_time = time.time() - start_time
    
    # Resultados finales
    print("\n🎯 ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"✅ Mejor accuracy: {results['best_accuracy']:.2f}%")
    print(f"⏱️  Tiempo total: {training_time/3600:.1f} horas")
    print(f"📊 Épocas completadas: {results['final_epoch']}")
    print(f"💾 Modelo guardado: ./binary_models/best_binary_model.pth")
    
    return results

if __name__ == "__main__":
    results = main()