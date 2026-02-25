# !/usr/bin/env python3
"""
reentrenamiento selectivo de breeds problematicas of the model resnet50
this script permite reentrenar only ciertas breeds that muestran baja precision
for mejorar su rendimiento without afectar el resto of the model
"""

# imports of the system operativo y manejo de files
import os         # operaciones of the system operativo
import shutil     # operaciones avanzadas de files
from pathlib import Path  # manejo moderno de rutas de files
import random     # generacion de numeros aleatorios

# imports de pytorch for deep learning
import torch              # framework principal de deep learning
import torch.nn as nn     # modulos de redes neuronales
import torch.optim as optim  # optimizadores for training
from torch.utils.data import Dataset, DataLoader  # manejo de datasets

# imports de computer vision
from torchvision import transforms, models  # transformaciones y models preentrenados
from PIL import Image     # processing de images

# dataset personalizado for load only breeds especificas seleccionadas
# permite entrenar de forma selectiva only las breeds that necesitan mejorarse
class SelectiveBreedDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_breeds=None):
        """
inicializa dataset that load only breeds objetivo especificadas
        
parametros:
- data_dir: directory base with subdirectorios de breeds
- transform: transformaciones a aplicar a las images
- target_breeds: list de names de breeds a incluir
        """
        self.data_dir = data_dir           # directory raiz de data
        self.transform = transform         # transformaciones de images
        self.target_breeds = target_breeds or []  # breeds objetivo with fallback
        
        self.samples = []                  # list de tuplas image, etiqueta
        self.class_to_idx = {}            # mapping de name de breed a index numerico
        
        # filtra only directorios that corresponden a breeds objetivo
        # verifica that sean directorios validos y esten en target_breeds
        available_breeds = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d)) and d in target_breeds]
        
        # construye mapping de classes y recolecta all las images
        for idx, breed in enumerate(available_breeds):
            self.class_to_idx[breed] = idx    # asigna index numerico a cada breed
            breed_path = os.path.join(data_dir, breed)
            
            # busca all los files de image en el directory de la breed
            for img_file in os.listdir(breed_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # agrega tupla de ruta de image y su etiqueta numerica
                    self.samples.append((os.path.join(breed_path, img_file), idx))
        
        # muestra estadisticas of the dataset creado for verificacion
        print(f"📊 Dataset creado: {len(self.samples)} imágenes, {len(available_breeds)} clases")
        for breed, idx in self.class_to_idx.items():
            count = sum(1 for s in self.samples if s[1] == idx)
            print(f"   {idx}: {breed} - {count} imágenes")
    
    # method requerido por pytorch dataset that devuelve numero total de muestras
    # permite that pytorch sepa cuantas iteraciones hacer en cada epoch
    def __len__(self):
        return len(self.samples)
    
    # method requerido por pytorch dataset that devuelve una muestra especifica
    # se llama automaticamente durante el training for obtener cada image
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]  # obtiene ruta y etiqueta of the index
        
        try:
            # load image y convierte a rgb for garantizar 3 canales
            image = Image.open(img_path).convert('RGB')
            
            # aplica transformaciones if fueron especificadas
            if self.transform:
                image = self.transform(image)
                
            return image, label  # devuelve tensor de image y etiqueta
            
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            # estrategia de recuperacion: devuelve image diferente valida
            # evita that el training se detenga por images corruptas
            return self.__getitem__((idx + 1) % len(self.samples))

# model de red neuronal for classification selectiva de breeds
# usa arquitectura resnet34 mas liviana for reentrenamiento fast
class SelectiveBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        """
inicializa clasificador with numero especifico de classes
usa resnet34 como backbone por ser mas fast that resnet50
        """
        super().__init__()
        
        # load resnet34 without pesos preentrenados for empezar desde cero
        self.backbone = models.resnet34(weights=None)
        
        # reemplaza la capa final for that coincida with numero de breeds objetivo
        # in_features obtiene el numero de neuronas de la capa anterior
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    # method forward define como fluyen los data a traves de la red
    def forward(self, x):
        return self.backbone(x)  # pasa data a traves de resnet34

def create_selective_fine_tuning():
    print("🔧 REENTRENAMIENTO SELECTIVO - RAZAS PROBLEMÁTICAS")
    print("=" * 70)
    
    # Implementation note.
    target_breeds = [
        'Labrador_retriever',
        'Norwegian_elkhound', 
        'beagle',
        'pug',
        'basset',
        'Samoyed'  # Implementation note.
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Dispositivo: {device}")
    
    # Transformaciones de training with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    train_dataset = SelectiveBreedDataset(
        'breed_processed_data/train', 
        transform=train_transform,
        target_breeds=target_breeds
    )
    
    # Crear directory de validation if no existe
    val_dir = 'breed_processed_data/val'
    if not os.path.exists(val_dir):
        print("📁 Creando dataset de validación...")
        os.makedirs(val_dir, exist_ok=True)
        
        # Mover 20% de images a validation
        for breed in target_breeds:
            breed_train = f'breed_processed_data/train/{breed}'
            breed_val = f'{val_dir}/{breed}'
            
            if os.path.exists(breed_train):
                os.makedirs(breed_val, exist_ok=True)
                
                images = [f for f in os.listdir(breed_train) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Mover 20% aleatoriamente
                val_count = max(1, len(images) // 5)
                val_images = random.sample(images, val_count)
                
                for img in val_images:
                    src = os.path.join(breed_train, img)
                    dst = os.path.join(breed_val, img)
                    shutil.move(src, dst)
                
                print(f"   {breed}: {len(val_images)} → validación")
    
    val_dataset = SelectiveBreedDataset(
        val_dir,
        transform=val_transform,
        target_breeds=target_breeds
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Model
    num_classes = len(train_dataset.class_to_idx)
    model = SelectiveBreedClassifier(num_classes).to(device)
    
    # load model preentrenado como punto de partida
    pretrained_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    if os.path.exists(pretrained_path):
        print("🔄 Cargando modelo preentrenado como punto de partida...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Only load pesos of the backbone (without la capa final)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filtrar only pesos of the backbone
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'fc' not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("✅ Backbone cargado, reentrenando clasificador")
    
    # Configuration de training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training
    print(f"\n🚀 Iniciando fine-tuning ({num_classes} clases)...")
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(20):  # Implementation note.
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/20, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/20:")
        print(f"  Train: Loss {train_loss/len(train_loader):.4f}, Acc {train_acc:.2f}%")
        print(f"  Val:   Loss {val_loss/len(val_loader):.4f}, Acc {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs('selective_models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'class_to_idx': train_dataset.class_to_idx,
                'target_breeds': target_breeds
            }, 'selective_models/best_selective_model.pth')
            
            print(f"✅ Nuevo mejor modelo guardado: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print("⏹️ Early stopping")
            break
    
    print(f"\n🏆 Mejor accuracy de validación: {best_val_acc:.2f}%")
    return 'selective_models/best_selective_model.pth', train_dataset.class_to_idx

if __name__ == "__main__":
    model_path, class_mapping = create_selective_fine_tuning()
    print(f"✅ Modelo selectivo guardado: {model_path}")
    print(f"📋 Clases: {list(class_mapping.keys())}")