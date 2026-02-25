"""
Preprocesador de datos para clasificación binaria PERRO vs NO-PERRO
Optimizado para entrenamiento con GPU AMD 7900XTX
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import json
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class DogClassificationDataset(Dataset):
    """Dataset personalizado para clasificación binaria de perros"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Cargar imagen
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error cargando imagen {image_path}: {e}")
            # Imagen de fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, torch.tensor(label, dtype=torch.float32)

class DataPreprocessor:
    """Preprocesador principal para el dataset"""
    
    def __init__(self, dataset_path: str, output_path: str, target_size: tuple = (224, 224)):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.yesdog_path = self.dataset_path / "YESDOG"
        self.nodog_path = self.dataset_path / "NODOG"
        
        # Crear directorio de salida
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar extensiones de imagen válidas
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Estadísticas para normalización (ImageNet)
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
    def collect_all_images(self):
        """Recolecta todas las rutas de imágenes y etiquetas"""
        print("📂 Recolectando rutas de imágenes...")
        
        image_paths = []
        labels = []
        
        # Imágenes de perros (etiqueta 1)
        print("   Procesando imágenes de perros...")
        dog_count = 0
        for breed_folder in tqdm(list(self.yesdog_path.iterdir())):
            if breed_folder.is_dir():
                for img_file in breed_folder.iterdir():
                    if img_file.suffix.lower() in self.image_extensions:
                        if self._is_valid_image(img_file):
                            image_paths.append(img_file)
                            labels.append(1)  # Perro
                            dog_count += 1
        
        # Imágenes de no-perros (etiqueta 0)
        print("   Procesando imágenes de no-perros...")
        nodog_count = 0
        for category_folder in tqdm(list(self.nodog_path.iterdir())):
            if category_folder.is_dir():
                for img_file in category_folder.iterdir():
                    if img_file.suffix.lower() in self.image_extensions:
                        if self._is_valid_image(img_file):
                            image_paths.append(img_file)
                            labels.append(0)  # No-perro
                            nodog_count += 1
        
        print(f"✅ Recolección completada:")
        print(f"   - Imágenes de perros: {dog_count:,}")
        print(f"   - Imágenes de no-perros: {nodog_count:,}")
        print(f"   - Total: {len(image_paths):,}")
        print(f"   - Ratio perros/no-perros: {dog_count/max(nodog_count, 1):.2f}")
        
        return image_paths, labels
    
    def _is_valid_image(self, image_path: Path) -> bool:
        """Verifica si una imagen es válida"""
        try:
            img = cv2.imread(str(image_path))
            return img is not None and img.shape[0] > 0 and img.shape[1] > 0
        except:
            return False
    
    def balance_classes(self, image_paths: list, labels: list, strategy: str = 'undersample'):
        """Balancea las clases del dataset"""
        print(f"⚖️ Balanceando clases con estrategia: {strategy}")
        
        # Separar por clases
        dog_indices = [i for i, label in enumerate(labels) if label == 1]
        nodog_indices = [i for i, label in enumerate(labels) if label == 0]
        
        dog_count = len(dog_indices)
        nodog_count = len(nodog_indices)
        
        print(f"   Antes - Perros: {dog_count:,}, No-perros: {nodog_count:,}")
        
        if strategy == 'undersample':
            # Reducir la clase mayoritaria
            target_size = min(dog_count, nodog_count)
            
            if dog_count > target_size:
                dog_indices = random.sample(dog_indices, target_size)
            if nodog_count > target_size:
                nodog_indices = random.sample(nodog_indices, target_size)
                
        elif strategy == 'oversample':
            # Aumentar la clase minoritaria (duplicando imágenes)
            target_size = max(dog_count, nodog_count)
            
            if dog_count < target_size:
                needed = target_size - dog_count
                dog_indices.extend(random.choices(dog_indices, k=needed))
            if nodog_count < target_size:
                needed = target_size - nodog_count
                nodog_indices.extend(random.choices(nodog_indices, k=needed))
        
        # Reconstruir listas balanceadas
        balanced_indices = dog_indices + nodog_indices
        balanced_paths = [image_paths[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        # Mezclar
        combined = list(zip(balanced_paths, balanced_labels))
        random.shuffle(combined)
        balanced_paths, balanced_labels = zip(*combined)
        
        print(f"   Después - Perros: {balanced_labels.count(1):,}, No-perros: {balanced_labels.count(0):,}")
        
        return list(balanced_paths), list(balanced_labels)
    
    def create_train_val_test_split(self, image_paths: list, labels: list, 
                                  train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Divide el dataset en train/validation/test"""
        print(f"📊 Dividiendo dataset: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%}")
        
        # Primera división: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=(1-train_ratio), 
            random_state=42, stratify=labels
        )
        
        # Segunda división: val vs test
        val_size = val_ratio / (val_ratio + (1-train_ratio-val_ratio))
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=(1-val_size),
            random_state=42, stratify=temp_labels
        )
        
        splits = {
            'train': {'paths': train_paths, 'labels': train_labels},
            'val': {'paths': val_paths, 'labels': val_labels},
            'test': {'paths': test_paths, 'labels': test_labels}
        }
        
        for split_name, split_data in splits.items():
            dog_count = split_data['labels'].count(1)
            nodog_count = split_data['labels'].count(0)
            total = len(split_data['labels'])
            print(f"   {split_name.upper():5s}: {total:5,} imágenes (perros: {dog_count:,}, no-perros: {nodog_count:,})")
        
        return splits
    
    def get_augmentation_transforms(self, mode: str = 'train'):
        """Define las transformaciones de augmentación"""
        
        if mode == 'train':
            # Augmentación agresiva para entrenamiento
            transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.RandomCrop(height=int(self.target_size[0]*0.9), 
                           width=int(self.target_size[1]*0.9), p=0.3),
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.2),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
                ToTensorV2()
            ])
        else:
            # Solo transformaciones básicas para validación y test
            transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
                ToTensorV2()
            ])
        
        return transform
    
    def create_data_loaders(self, splits: dict, batch_size: int = 32, num_workers: int = 4):
        """Crea los DataLoaders para entrenamiento"""
        print(f"🔄 Creando DataLoaders (batch_size={batch_size}, num_workers={num_workers})...")
        
        # Transformaciones
        train_transform = self.get_augmentation_transforms('train')
        val_transform = self.get_augmentation_transforms('val')
        
        # Datasets
        train_dataset = DogClassificationDataset(
            splits['train']['paths'], 
            splits['train']['labels'], 
            transform=train_transform
        )
        
        val_dataset = DogClassificationDataset(
            splits['val']['paths'], 
            splits['val']['labels'], 
            transform=val_transform
        )
        
        test_dataset = DogClassificationDataset(
            splits['test']['paths'], 
            splits['test']['labels'], 
            transform=val_transform
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✅ DataLoaders creados:")
        print(f"   - Train: {len(train_loader)} batches")
        print(f"   - Val:   {len(val_loader)} batches")
        print(f"   - Test:  {len(test_loader)} batches")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def save_preprocessing_info(self, splits: dict):
        """Guarda información del preprocesamiento"""
        info = {
            'dataset_stats': {
                'total_images': sum(len(split['labels']) for split in splits.values()),
                'train_images': len(splits['train']['labels']),
                'val_images': len(splits['val']['labels']),
                'test_images': len(splits['test']['labels']),
            },
            'class_distribution': {
                'train': {
                    'dogs': splits['train']['labels'].count(1),
                    'no_dogs': splits['train']['labels'].count(0)
                },
                'val': {
                    'dogs': splits['val']['labels'].count(1),
                    'no_dogs': splits['val']['labels'].count(0)
                },
                'test': {
                    'dogs': splits['test']['labels'].count(1),
                    'no_dogs': splits['test']['labels'].count(0)
                }
            },
            'preprocessing_config': {
                'target_size': self.target_size,
                'normalization_mean': self.imagenet_mean,
                'normalization_std': self.imagenet_std,
                'augmentation_enabled': True
            }
        }
        
        info_path = self.output_path / 'preprocessing_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"💾 Información guardada en: {info_path}")
        
    def process_complete_dataset(self, balance_strategy: str = 'undersample', 
                               batch_size: int = 32):
        """Procesa el dataset completo"""
        print("🚀 Iniciando preprocesamiento completo...")
        print("="*60)
        
        # 1. Recolectar imágenes
        image_paths, labels = self.collect_all_images()
        
        # 2. Balancear clases
        if balance_strategy:
            image_paths, labels = self.balance_classes(image_paths, labels, balance_strategy)
        
        # 3. Dividir en train/val/test
        splits = self.create_train_val_test_split(image_paths, labels)
        
        # 4. Crear DataLoaders
        data_loaders = self.create_data_loaders(splits, batch_size=batch_size)
        
        # 5. Guardar información
        self.save_preprocessing_info(splits)
        
        print("\n🎉 ¡Preprocesamiento completado exitosamente!")
        return data_loaders, splits

def create_sample_visualization(data_loaders, save_path: str):
    """Crea visualización de muestras del dataset"""
    import matplotlib.pyplot as plt
    
    print("📸 Creando visualización de muestras...")
    
    # Obtener batch de entrenamiento
    train_loader = data_loaders['train']
    batch_iter = iter(train_loader)
    images, labels = next(batch_iter)
    
    # Desnormalizar para visualización
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Muestras del Dataset con Augmentación', fontsize=16)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # Desnormalizar imagen
        img = images[i].clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = torch.clamp(img, 0, 1)
        
        # Convertir a numpy
        img_np = img.permute(1, 2, 0).numpy()
        
        # Mostrar
        axes[row, col].imshow(img_np)
        label_text = "🐕 PERRO" if labels[i].item() == 1 else "📦 NO-PERRO"
        axes[row, col].set_title(label_text, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualización guardada en: {save_path}")

if __name__ == "__main__":
    # Configuración
    dataset_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS"
    output_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\processed_data"
    
    # Crear preprocesador
    preprocessor = DataPreprocessor(dataset_path, output_path)
    
    # Procesar dataset
    data_loaders, splits = preprocessor.process_complete_dataset(
        balance_strategy='undersample',  # 'undersample', 'oversample', or None
        batch_size=32
    )
    
    # Crear visualización
    sample_viz_path = Path(output_path) / 'sample_visualization.png'
    create_sample_visualization(data_loaders, str(sample_viz_path))