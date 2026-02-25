"""
🐕 ENTRENADOR BINARIO SIMPLIFICADO And ESTABLE
Technical documentation in English.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar environment antes of importar PyTorch
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

# Control of parada simplificado
class SimpleController:
    def __init__(self):
        self.should_stop = False
    
    def check_for_stop(self):
        """Technical documentation in English."""
        try:
            import select
            import sys
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                input_line = sys.stdin.readline()
                if 'q' in input_line.lower():
                    self.should_stop = True
                    return True
        except:
            pass
        return False

class SimpleBinaryDataset(Dataset):
    """Dataset binario simplificado"""
    
    def __init__(self, data_path, transform=None, max_per_class=10000):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        self._load_samples(max_per_class)
        
    def _load_samples(self, max_per_class):
        print("📊 Cargando dataset binario simplificado...")
        
        # NO-dog
        nodog_path = self.data_path / "NODOG"
        no_dog_count = 0
        if nodog_path.exists():
            # Load files directos
            for img_file in nodog_path.glob("*.jpg"):
                if no_dog_count >= max_per_class:
                    break
                self.samples.append((str(img_file), 0))
                no_dog_count += 1
            
            # Load subdirectories
            for subdir in nodog_path.iterdir():
                if subdir.is_dir() and no_dog_count < max_per_class:
                    for img_file in subdir.glob("*.jpg"):
                        if no_dog_count >= max_per_class:
                            break
                        self.samples.append((str(img_file), 0))
                        no_dog_count += 1
        
        print(f"   ❌ NO-PERRO: {no_dog_count:,} imágenes")
        
        # dog
        yesdog_path = self.data_path / "YESDOG"
        dog_count = 0
        if yesdog_path.exists():
            for breed_dir in yesdog_path.iterdir():
                if breed_dir.is_dir() and dog_count < max_per_class:
                    for img_file in list(breed_dir.glob("*.JPEG")) + list(breed_dir.glob("*.jpg")):
                        if dog_count >= max_per_class:
                            break
                        self.samples.append((str(img_file), 1))
                        dog_count += 1
        
        print(f"   ✅ PERRO: {dog_count:,} imágenes")
        print(f"   🎯 Total: {len(self.samples):,} muestras")
        
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
            # If falla a image, return a image en blanco
            print(f"⚠️ Error cargando {img_path}: {e}")
            blank_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, label

class SimpleBinaryModel(nn.Module):
    """Model binario simplificado usando ResNet18"""
    
    def __init__(self):
        super().__init__()
        # Use ResNet18 en lugar of EfficientNet for evitar problemas
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 2)  # 2 classes: dog/no-dog
        
    def forward(self, x):
        return self.backbone(x)

class SimpleTrainer:
    """Entrenador simplificado"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.controller = SimpleController()
        
        # Configuration simple
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, train_loader, epoch):
        """Technical documentation in English."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Época {epoch}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Verify parada cada 10 batches
            if batch_idx % 10 == 0 and self.controller.check_for_stop():
                print("\n🛑 Parada solicitada por usuario")
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Implementation note.
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update bar
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Valid the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train_model(self, train_loader, val_loader, epochs=20, save_path='./binary_models'):
        """Training complete"""
        print("🚀 ENTRENAMIENTO BINARIO SIMPLIFICADO")
        print("=" * 60)
        print(f"🎯 Épocas: {epochs}")
        print(f"💻 Dispositivo: {self.device}")
        print("⚠️  Presiona Enter + 'q' + Enter para parar")
        print()
        
        Path(save_path).mkdir(exist_ok=True)
        best_val_acc = 0
        
        for epoch in range(1, epochs + 1):
            if self.controller.should_stop:
                print(f"🛑 Entrenamiento detenido en época {epoch}")
                break
                
            print(f"📅 ÉPOCA {epoch}/{epochs}")
            print("-" * 40)
            
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validar
            val_loss, val_acc = self.validate(val_loader)
            
            # Implementation note.
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Show resultados
            print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = Path(save_path) / 'best_binary_model.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': val_acc,
                    'epoch': epoch,
                }, model_path)
                print(f"✅ Mejor modelo guardado: {val_acc:.2f}%")
            
            print()
        
        print(f"🎯 MEJOR ACCURACY ALCANZADA: {best_val_acc:.2f}%")
        return {'best_accuracy': best_val_acc}

def get_simple_transforms():
    """Transformaciones simplificadas"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    """Function main"""
    print("🐕 ENTRENADOR BINARIO SIMPLIFICADO")
    print("🚀 Versión estable sin conflictos de PyTorch")
    print("=" * 80)
    
    # Configuration
    DATA_PATH = "./DATASETS"
    BATCH_SIZE = 16  # Implementation note.
    EPOCHS = 20
    MAX_PER_CLASS = 10000  # 10k for class
    
    # Verify data
    if not Path(DATA_PATH).exists():
        print(f"❌ Directorio de datos no encontrado: {DATA_PATH}")
        return
    
    # Transformaciones
    train_transform, val_transform = get_simple_transforms()
    
    # Datasets
    print("📊 Creando datasets...")
    train_dataset = SimpleBinaryDataset(DATA_PATH, train_transform, MAX_PER_CLASS)
    val_dataset = SimpleBinaryDataset(DATA_PATH, val_transform, MAX_PER_CLASS//3)
    
    # DataLoaders simplificados
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✅ Train samples: {len(train_dataset):,}")
    print(f"✅ Val samples: {len(val_dataset):,}")
    print()
    
    # Model
    print("🤖 Creando modelo ResNet18...")
    model = SimpleBinaryModel()
    device = torch.device('cpu')
    
    # Trainer
    trainer = SimpleTrainer(model, device)
    
    # Entrenar
    results = trainer.train_model(train_loader, val_loader, EPOCHS)
    
    print("🎉 ENTRENAMIENTO COMPLETADO!")
    print(f"✅ Mejor accuracy: {results['best_accuracy']:.2f}%")
    print(f"💾 Modelo guardado en: binary_models/best_binary_model.pth")
    
    # Implementation note.
    import shutil
    src = "binary_models/best_binary_model.pth"
    dst = "best_model.pth"
    if Path(src).exists():
        shutil.copy2(src, dst)
        print(f"📋 Modelo copiado a: {dst}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()