"""
Optimizador of inferencia for model of classification dog vs NO-dog
Technical documentation in English.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import time
import json
from typing import Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class InferenceOptimizer:
    """Optimizador of inferencia for models of classification"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = Path(model_path)
        
        # Configurar device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # load model original
        self.model = self._load_model()
        self.model.eval()
        
        # Transformaciones of preprocesamiento
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Models optimizados
        self.torchscript_model = None
        self.onnx_session = None
        
        print(f"🚀 InferenceOptimizer inicializado")
        print(f"   Dispositivo: {self.device}")
        print(f"   Modelo cargado desde: {self.model_path}")
    
    def _load_model(self):
        """Load the model entrenado"""
        from model_trainer import DogClassificationModel
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with the misma arquitectura
        model_name = checkpoint.get('model_name', 'efficientnet_b3')
        model = DogClassificationModel(model_name=model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def optimize_to_torchscript(self, save_path: str = None):
        """Convierte the model a TorchScript for inferencia optimizada"""
        print("🔧 Optimizando modelo a TorchScript...")
        
        # Preparar ejemplo of input
        example_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Tracing method
        try:
            self.torchscript_model = torch.jit.trace(self.model, example_input)
            print("✅ Modelo convertido usando torch.jit.trace")
        except Exception as e:
            print(f"⚠️  Error con trace, intentando script: {e}")
            try:
                # Scripting method
                self.torchscript_model = torch.jit.script(self.model)
                print("✅ Modelo convertido usando torch.jit.script")
            except Exception as e2:
                print(f"❌ Error convirtiendo a TorchScript: {e2}")
                return None
        
        # Optimizaciones adicionales
        if self.torchscript_model:
            # Freeze for inference
            self.torchscript_model = torch.jit.freeze(self.torchscript_model)
            
            # Optimizar for inferencia
            self.torchscript_model = torch.jit.optimize_for_inference(self.torchscript_model)
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                self.torchscript_model.save(str(save_path))
                print(f"💾 Modelo TorchScript guardado: {save_path}")
        
        return self.torchscript_model
    
    def optimize_to_onnx(self, save_path: str = None):
        """Technical documentation in English."""
        print("🔧 Optimizando modelo a ONNX...")
        
        if save_path is None:
            save_path = self.model_path.parent / "model_optimized.onnx"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Preparar input of ejemplo
        example_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        try:
            # Exportar a ONNX
            torch.onnx.export(
                self.model,
                example_input,
                str(save_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify model ONNX
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)
            
            # Implementation note.
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(str(save_path), providers=providers)
            
            print(f"✅ Modelo ONNX creado y verificado")
            print(f"💾 Guardado en: {save_path}")
            print(f"   Providers: {self.onnx_session.get_providers()}")
            
            return str(save_path)
            
        except Exception as e:
            print(f"❌ Error convirtiendo a ONNX: {e}")
            return None
    
    def benchmark_models(self, num_runs: int = 100, batch_sizes: List[int] = [1, 4, 8, 16]):
        """Benchmarks of velocidad for diferentes optimizaciones"""
        print(f"⏱️  Ejecutando benchmark con {num_runs} runs...")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Preparar data of test
            test_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
            test_input_np = test_input.cpu().numpy()
            
            # Benchmark model original
            results[f'pytorch_b{batch_size}'] = self._benchmark_pytorch(test_input, num_runs)
            
            # Benchmark TorchScript
            if self.torchscript_model:
                results[f'torchscript_b{batch_size}'] = self._benchmark_torchscript(test_input, num_runs)
            
            # Benchmark ONNX
            if self.onnx_session:
                results[f'onnx_b{batch_size}'] = self._benchmark_onnx(test_input_np, num_runs)
        
        # Show resultados
        self._print_benchmark_results(results)
        
        return results
    
    def _benchmark_pytorch(self, input_tensor: torch.Tensor, num_runs: int) -> dict:
        """Benchmark of the model PyTorch original"""
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = self.model(input_tensor)
            
            # Implementation note.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Medir time
            start_time = time.time()
            for _ in range(num_runs):
                output = self.model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            fps = 1.0 / avg_time * input_tensor.shape[0]
            
            return {
                'avg_time_ms': avg_time * 1000,
                'fps': fps,
                'total_time': total_time
            }
    
    def _benchmark_torchscript(self, input_tensor: torch.Tensor, num_runs: int) -> dict:
        """Benchmark of the model TorchScript"""
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = self.torchscript_model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(num_runs):
                output = self.torchscript_model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            fps = 1.0 / avg_time * input_tensor.shape[0]
            
            return {
                'avg_time_ms': avg_time * 1000,
                'fps': fps,
                'total_time': total_time
            }
    
    def _benchmark_onnx(self, input_array: np.ndarray, num_runs: int) -> dict:
        """Benchmark of the model ONNX"""
        # Warmup
        for _ in range(10):
            _ = self.onnx_session.run(None, {'input': input_array})
        
        start_time = time.time()
        for _ in range(num_runs):
            output = self.onnx_session.run(None, {'input': input_array})
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time * input_array.shape[0]
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'total_time': total_time
        }
    
    def _print_benchmark_results(self, results: dict):
        """Imprime resultados of the benchmark"""
        print("\n📊 RESULTADOS DEL BENCHMARK")
        print("="*60)
        
        for key, metrics in results.items():
            print(f"{key:20s}: {metrics['avg_time_ms']:6.2f} ms, {metrics['fps']:6.1f} FPS")
        
        # Implementation note.
        fastest_key = min(results.keys(), key=lambda k: results[k]['avg_time_ms'])
        print(f"\n🏆 Más rápido: {fastest_key}")
    
    def create_production_model(self, format: str = 'torchscript', save_dir: str = './optimized_models'):
        """Technical documentation in English."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏭 Creando modelo de producción ({format})...")
        
        if format == 'torchscript':
            model_path = save_dir / 'production_model.pt'
            self.optimize_to_torchscript(str(model_path))
            
            # Create metadata
            metadata = {
                'format': 'torchscript',
                'model_path': str(model_path),
                'input_shape': [1, 3, 224, 224],
                'output_shape': [1],
                'preprocessing': {
                    'resize': [224, 224],
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225]
                }
            }
            
        elif format == 'onnx':
            model_path = save_dir / 'production_model.onnx'
            self.optimize_to_onnx(str(model_path))
            
            metadata = {
                'format': 'onnx',
                'model_path': str(model_path),
                'input_shape': [1, 3, 224, 224],
                'output_shape': [1],
                'preprocessing': {
                    'resize': [224, 224],
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225]
                }
            }
        
        # Save metadata
        metadata_path = save_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Modelo de producción creado:")
        print(f"   Formato: {format}")
        print(f"   Modelo: {model_path}")
        print(f"   Metadata: {metadata_path}")
        
        return str(model_path), str(metadata_path)

class ProductionInference:
    """Technical documentation in English."""
    
    def __init__(self, model_path: str, metadata_path: str = None):
        self.model_path = Path(model_path)
        
        # Load metadata
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Metadata by default
            self.metadata = {
                'preprocessing': {
                    'resize': [224, 224],
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225]
                }
            }
        
        # Implementation note.
        self.format = self.metadata.get('format', 'torchscript')
        self._load_model()
        
        print(f"🚀 ProductionInference inicializado ({self.format})")
    
    def _load_model(self):
        """Load the model optimized"""
        if self.format == 'torchscript':
            self.model = torch.jit.load(str(self.model_path))
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
        elif self.format == 'onnx':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(str(self.model_path), providers=providers)
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Preprocesa image for inferencia"""
        # Load image if es a path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        target_size = self.metadata['preprocessing']['resize']
        image = cv2.resize(image, tuple(target_size))
        
        # Normalizar
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.metadata['preprocessing']['normalize_mean'])
        std = np.array(self.metadata['preprocessing']['normalize_std'])
        image = (image - mean) / std
        
        # Reorganizar dimensiones: HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Implementation note.
        image = np.expand_dims(image, axis=0)
        
        if self.format == 'torchscript':
            return torch.from_numpy(image).to(self.device)
        else:
            return image.astype(np.float32)
    
    def predict(self, image: Union[str, np.ndarray]) -> Tuple[float, str]:
        """Realiza prediction en a image"""
        # Preprocesar
        input_tensor = self.preprocess_image(image)
        
        # Inferencia
        if self.format == 'torchscript':
            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).item()
        else:
            output = self.model.run(None, {'input': input_tensor})[0]
            probability = 1.0 / (1.0 + np.exp(-output[0]))  # sigmoid
        
        # Clasificar
        is_dog = probability > 0.5
        label = "🐕 PERRO" if is_dog else "📦 NO-PERRO"
        
        return probability, label
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Tuple[float, str]]:
        """Realiza prediction en a lote of images"""
        results = []
        
        # Preprocesar all the images
        batch_inputs = []
        for image in images:
            input_tensor = self.preprocess_image(image)
            if self.format == 'torchscript':
                batch_inputs.append(input_tensor)
            else:
                batch_inputs.append(input_tensor[0])  # Implementation note.
        
        if self.format == 'torchscript':
            # Concatenar en batch
            batch_tensor = torch.cat(batch_inputs, dim=0)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
        else:
            # ONNX batch processing
            batch_array = np.stack(batch_inputs, axis=0)
            outputs = self.model.run(None, {'input': batch_array})[0]
            probabilities = 1.0 / (1.0 + np.exp(-outputs))  # sigmoid
        
        # Procesar resultados
        for prob in probabilities:
            if isinstance(prob, np.ndarray):
                prob = prob.item()
            is_dog = prob > 0.5
            label = "🐕 PERRO" if is_dog else "📦 NO-PERRO"
            results.append((prob, label))
        
        return results

if __name__ == "__main__":
    # Ejemplo of uso
    model_path = "path/to/your/best_model.pth"
    
    print("🚀 Iniciando optimización de inferencia...")
    
    # Create optimizador
    optimizer = InferenceOptimizer(model_path)
    
    # Optimizar a TorchScript
    optimizer.optimize_to_torchscript()
    
    # Optimizar a ONNX
    optimizer.optimize_to_onnx()
    
    # Benchmark
    results = optimizer.benchmark_models(num_runs=50)
    
    # Implementation note.
    prod_model_path, metadata_path = optimizer.create_production_model('torchscript')
    
    # Implementation note.
    inference = ProductionInference(prod_model_path, metadata_path)
    
    # Prediction of ejemplo (necesita a image real)
    # probability, label = inference.predict("path/to/test/image.jpg")
    # print(f"Resultado: {label} (probabilidad: {probability:.3f})")