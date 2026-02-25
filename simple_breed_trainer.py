"""
🐕 ENTRENADOR of breeds SIMPLIFICADO And ESTABLE
Technical documentation in English.
"""

import os
import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar environment
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

class SimpleBreedDataset(Dataset):
    """Dataset of breeds simplificado"""
    
    def __init__(self, data_path="./breed_processed_data", split="train", transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        
        # Load configuration
        with open(self.data_path / "dataset_info.json", 'r') as f:
            self.info = json.load(f)
        
        self.num_classes = self.info['total_breeds']
        self._load_split_samples(split)
        
    def _load_split_samples(self, split):
        """Load samples of the split especificado from folders"""
        print(f"📂 Cargando {split} split...")
        
        split_dir = self.data_path / split
        if not split_dir.exists():
            print(f"❌ Directorio no encontrado: {split_dir}")
            return
        
        # Load samples of cada folder of class
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Encontrar the index of class
                class_idx = None
                for breed, info in self.info['breed_details'].items():
                    if info['display_name'].lower().replace(' ', '_') == class_name.lower():
                        class_idx = info['class_index']
                        break
                
                if class_idx is None:
                    # Search for name directo
                    if class_name in self.info['breed_details']:
                        class_idx = self.info['breed_details'][class_name]['class_index']
                    else:
                        print(f"⚠️ Clase no encontrada: {class_name}")
                        continue
                
                # Load images of this class
                for img_file in class_dir.glob("*.JPEG"):
                    self.samples.append((str(img_file), class_idx))
                for img_file in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_file), class_idx))
        
        print(f"   ✅ {len(self.samples):,} muestras cargadas")
        
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
            print(f"⚠️ Error cargando {img_path}: {e}")
            # Return image en blanco if falla
            blank_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, label

class SimpleBreedModel(nn.Module):
    """Model of breeds simplificado usando ResNet34"""
    
    def __init__(self, num_classes=50):
        super().__init__()
        # Implementation note.
        self.backbone = models.resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class SimpleBreedTrainer:
    """Entrenador of breeds simplificado"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Configuration simple pero efectiva
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
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
            
            # Update bar cada 10 batches
            if batch_idx % 10 == 0:
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
            for data, target in tqdm(val_loader, desc="Validando"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train_model(self, train_loader, val_loader, epochs=25, save_path='./breed_models'):
        """Training complete"""
        print("🐕 ENTRENAMIENTO DE RAZAS SIMPLIFICADO")
        print("=" * 60)
        print(f"🎯 Épocas: {epochs}")
        print(f"🏷️ Clases: {self.model.backbone.fc.out_features}")
        print(f"💻 Dispositivo: {self.device}")
        print()
        
        Path(save_path).mkdir(exist_ok=True)
        best_val_acc = 0
        patience_counter = 0
        patience = 5
        
        for epoch in range(1, epochs + 1):
            print(f"📅 ÉPOCA {epoch}/{epochs}")
            print("-" * 40)
            
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validar
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Implementation note.
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Show resultados
            print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"🔄 Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                model_path = Path(save_path) / 'best_breed_model_simple.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': val_acc,
                    'epoch': epoch,
                    'num_classes': self.model.backbone.fc.out_features
                }, model_path)
                print(f"✅ Mejor modelo guardado: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ Paciencia: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping en época {epoch}")
                break
            
            print()
        
        print(f"🎯 MEJOR ACCURACY ALCANZADA: {best_val_acc:.2f}%")
        return {'best_accuracy': best_val_acc, 'final_epoch': epoch}

def get_breed_transforms():
    """Transformaciones for breeds"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    print("🐕 ENTRENADOR DE RAZAS SIMPLIFICADO")
    print("🚀 50 Razas - Versión estable")
    print("=" * 80)
    
    # Configuration
    DATA_PATH = "./breed_processed_data"
    BATCH_SIZE = 16  # Conservativo for 50 classes
    EPOCHS = 25
    
    # Verify data procesados
    if not Path(DATA_PATH).exists():
        print(f"❌ Datos procesados no encontrados: {DATA_PATH}")
        print("💡 Ejecuta primero: python breed_preprocessor.py")
        return
    
    # Transformaciones
    train_transform, val_transform = get_breed_transforms()
    
    # Datasets
    print("📊 Creando datasets de razas...")
    train_dataset = SimpleBreedDataset(DATA_PATH, "train", train_transform)
    val_dataset = SimpleBreedDataset(DATA_PATH, "val", val_transform)
    
    print(f"✅ Train samples: {len(train_dataset):,}")
    print(f"✅ Val samples: {len(val_dataset):,}")
    print(f"🏷️ Número de razas: {train_dataset.num_classes}")
    print()
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model
    print("🤖 Creando modelo ResNet34 para 50 razas...")
    model = SimpleBreedModel(num_classes=train_dataset.num_classes)
    device = torch.device('cpu')
    
    # Trainer
    trainer = SimpleBreedTrainer(model, device)
    
    # Entrenar
    results = trainer.train_model(train_loader, val_loader, EPOCHS)
    
    print("🎉 ENTRENAMIENTO DE RAZAS COMPLETADO!")
    print(f"✅ Mejor accuracy: {results['best_accuracy']:.2f}%")
    print(f"📅 Épocas entrenadas: {results['final_epoch']}")
    print(f"💾 Modelo guardado en: breed_models/best_breed_model_simple.pth")
    
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