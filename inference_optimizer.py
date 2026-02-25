"""
Inference Optimizer Module
==========================

This module provides tools for optimizing deep learning model inference
for dog vs non-dog classification. It supports conversion to TorchScript
and ONNX formats for improved inference performance.

Key Features:
    - TorchScript model conversion with JIT tracing/scripting
    - ONNX export with dynamic batch size support
    - Comprehensive benchmarking across different optimizations
    - Production-ready model packaging with metadata
    - Support for both CPU and GPU (CUDA/ROCm) inference

Classes:
    InferenceOptimizer: Converts and benchmarks model optimizations.
    ProductionInference: High-performance inference engine for deployment.

Usage Example:
    >>> optimizer = InferenceOptimizer('path/to/model.pth')
    >>> optimizer.optimize_to_torchscript()
    >>> optimizer.optimize_to_onnx()
    >>> results = optimizer.benchmark_models(num_runs=100)

Author: Dog Classification Team
Version: 1.0.0
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
    """
    Inference optimizer for classification models.
    
    Provides functionality to convert trained PyTorch models to optimized
    formats (TorchScript, ONNX) and benchmark their performance.
    
    Attributes:
        model_path (Path): Path to the trained model checkpoint.
        device (torch.device): Computation device (CPU/CUDA).
        model: The loaded PyTorch model.
        transform: Image preprocessing transformations.
        torchscript_model: TorchScript optimized model (if converted).
        onnx_session: ONNX Runtime inference session (if converted).
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the inference optimizer.
        
        Args:
            model_path (str): Path to the trained model checkpoint (.pth file).
            device (str, optional): Computation device. Options: 'auto', 'cuda', 'cpu'.
                                   'auto' selects CUDA if available. Defaults to 'auto'.
        """
        self.model_path = Path(model_path)
        
        # Configure device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load original model
        self.model = self._load_model()
        self.model.eval()
        
        # Preprocessing transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Optimized models
        self.torchscript_model = None
        self.onnx_session = None
        
        print(f"InferenceOptimizer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model loaded from: {self.model_path}")
    
    def _load_model(self):
        """
        Load the trained model from checkpoint.
        
        Returns:
            DogClassificationModel: Loaded model with weights.
        
        Raises:
            FileNotFoundError: If model checkpoint doesn't exist.
        """
        from model_trainer import DogClassificationModel
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with same architecture
        model_name = checkpoint.get('model_name', 'efficientnet_b3')
        model = DogClassificationModel(model_name=model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def optimize_to_torchscript(self, save_path: str = None):
        """
        Convert the model to TorchScript for optimized inference.
        
        Attempts JIT tracing first, falling back to scripting if tracing fails.
        Applies additional optimizations including freezing and inference optimization.
        
        Args:
            save_path (str, optional): Path to save the TorchScript model.
                                       If None, model is not saved to disk.
        
        Returns:
            torch.jit.ScriptModule: The optimized TorchScript model, or None if conversion fails.
        """
        print("Optimizing model to TorchScript...")
        
        # Prepare example input
        example_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Try tracing method first
        try:
            self.torchscript_model = torch.jit.trace(self.model, example_input)
            print("Model converted using torch.jit.trace")
        except Exception as e:
            print(f"Warning: Trace failed, attempting script: {e}")
            try:
                # Fall back to scripting method
                self.torchscript_model = torch.jit.script(self.model)
                print("Model converted using torch.jit.script")
            except Exception as e2:
                print(f"Error converting to TorchScript: {e2}")
                return None
        
        # Apply additional optimizations
        if self.torchscript_model:
            # Freeze for inference
            self.torchscript_model = torch.jit.freeze(self.torchscript_model)
            
            # Optimize for inference
            self.torchscript_model = torch.jit.optimize_for_inference(self.torchscript_model)
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                self.torchscript_model.save(str(save_path))
                print(f"TorchScript model saved: {save_path}")
        
        return self.torchscript_model
    
    def optimize_to_onnx(self, save_path: str = None):
        """
        Convert the model to ONNX format for cross-platform inference.
        
        Exports the model with dynamic batch size support and creates an
        ONNX Runtime inference session for benchmarking.
        
        Args:
            save_path (str, optional): Path to save the ONNX model.
                                       Defaults to 'model_optimized.onnx' in model directory.
        
        Returns:
            str: Path to the saved ONNX model, or None if conversion fails.
        """
        print("Optimizing model to ONNX...")
        
        if save_path is None:
            save_path = self.model_path.parent / "model_optimized.onnx"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare example input
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
            
            # Verify ONNX model
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session for benchmarking
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(str(save_path), providers=providers)
            
            print(f"ONNX model created and verified")
            print(f"Saved to: {save_path}")
            print(f"   Providers: {self.onnx_session.get_providers()}")
            
            return str(save_path)
            
        except Exception as e:
            print(f"Error converting to ONNX: {e}")
            return None
    
    def benchmark_models(self, num_runs: int = 100, batch_sizes: List[int] = [1, 4, 8, 16]):
        """
        Benchmark inference speed across different model optimizations.
        
        Measures inference time and throughput (FPS) for PyTorch, TorchScript,
        and ONNX models across various batch sizes.
        
        Args:
            num_runs (int, optional): Number of inference runs per benchmark. Defaults to 100.
            batch_sizes (list, optional): List of batch sizes to test. Defaults to [1, 4, 8, 16].
        
        Returns:
            dict: Benchmark results with keys formatted as '{format}_b{batch_size}'.
                  Each entry contains 'avg_time_ms', 'fps', and 'total_time'.
        """
        print(f"Running benchmark with {num_runs} runs...")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Prepare test data
            test_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
            test_input_np = test_input.cpu().numpy()
            
            # Benchmark original PyTorch model
            results[f'pytorch_b{batch_size}'] = self._benchmark_pytorch(test_input, num_runs)
            
            # Benchmark TorchScript model
            if self.torchscript_model:
                results[f'torchscript_b{batch_size}'] = self._benchmark_torchscript(test_input, num_runs)
            
            # Benchmark ONNX model
            if self.onnx_session:
                results[f'onnx_b{batch_size}'] = self._benchmark_onnx(test_input_np, num_runs)
        
        # Display results
        self._print_benchmark_results(results)
        
        return results
    
    def _benchmark_pytorch(self, input_tensor: torch.Tensor, num_runs: int) -> dict:
        """
        Benchmark the original PyTorch model.
        
        Args:
            input_tensor (torch.Tensor): Input tensor for benchmarking.
            num_runs (int): Number of inference runs.
        
        Returns:
            dict: Benchmark metrics including avg_time_ms, fps, and total_time.
        """
        with torch.no_grad():
            # Warmup runs
            for _ in range(10):
                _ = self.model(input_tensor)
            
            # Synchronize CUDA for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Measure time
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
        """
        Benchmark the TorchScript optimized model.
        
        Args:
            input_tensor (torch.Tensor): Input tensor for benchmarking.
            num_runs (int): Number of inference runs.
        
        Returns:
            dict: Benchmark metrics including avg_time_ms, fps, and total_time.
        """
        with torch.no_grad():
            # Warmup runs
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
        """
        Benchmark the ONNX model using ONNX Runtime.
        
        Args:
            input_array (np.ndarray): Input array for benchmarking.
            num_runs (int): Number of inference runs.
        
        Returns:
            dict: Benchmark metrics including avg_time_ms, fps, and total_time.
        """
        # Warmup runs
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
        """
        Print formatted benchmark results.
        
        Args:
            results (dict): Dictionary containing benchmark metrics.
        """
        print("\nBENCHMARK RESULTS")
        print("="*60)
        
        for key, metrics in results.items():
            print(f"{key:20s}: {metrics['avg_time_ms']:6.2f} ms, {metrics['fps']:6.1f} FPS")
        
        # Identify fastest configuration
        fastest_key = min(results.keys(), key=lambda k: results[k]['avg_time_ms'])
        print(f"\nFastest: {fastest_key}")
    
    def create_production_model(self, format: str = 'torchscript', save_dir: str = './optimized_models'):
        """
        Create a production-ready model with metadata.
        
        Packages the optimized model along with preprocessing configuration
        and other metadata required for deployment.
        
        Args:
            format (str, optional): Model format ('torchscript' or 'onnx'). Defaults to 'torchscript'.
            save_dir (str, optional): Directory to save production model. Defaults to './optimized_models'.
        
        Returns:
            tuple: (model_path, metadata_path) paths to saved files.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating production model ({format})...")
        
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
        
        print(f"Production model created:")
        print(f"   Format: {format}")
        print(f"   Model: {model_path}")
        print(f"   Metadata: {metadata_path}")
        
        return str(model_path), str(metadata_path)


class ProductionInference:
    """
    High-performance inference engine for production deployment.
    
    Provides optimized inference using TorchScript or ONNX models with
    consistent preprocessing and batched prediction support.
    
    Attributes:
        model_path (Path): Path to the optimized model file.
        metadata (dict): Model metadata including preprocessing config.
        format (str): Model format ('torchscript' or 'onnx').
        model: The loaded inference model or ONNX session.
        device (torch.device): Computation device (for TorchScript).
    """
    
    def __init__(self, model_path: str, metadata_path: str = None):
        """
        Initialize the production inference engine.
        
        Args:
            model_path (str): Path to the optimized model file.
            metadata_path (str, optional): Path to model metadata JSON.
                                           If None, default preprocessing is used.
        """
        self.model_path = Path(model_path)
        
        # Load metadata
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Default metadata
            self.metadata = {
                'preprocessing': {
                    'resize': [224, 224],
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225]
                }
            }
        
        # Detect model format and load accordingly
        self.format = self.metadata.get('format', 'torchscript')
        self._load_model()
        
        print(f"ProductionInference initialized ({self.format})")
    
    def _load_model(self):
        """
        Load the optimized model based on format.
        
        Loads TorchScript models using torch.jit.load or creates
        an ONNX Runtime session for ONNX models.
        """
        if self.format == 'torchscript':
            self.model = torch.jit.load(str(self.model_path))
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
        elif self.format == 'onnx':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(str(self.model_path), providers=providers)
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Applies resizing, normalization, and format conversion based
        on the model's preprocessing requirements.
        
        Args:
            image: Input image as file path (str/Path) or numpy array (H, W, C).
        
        Returns:
            Preprocessed tensor (TorchScript) or array (ONNX) with shape (1, C, H, W).
        """
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target dimensions
        target_size = self.metadata['preprocessing']['resize']
        image = cv2.resize(image, tuple(target_size))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.metadata['preprocessing']['normalize_mean'])
        std = np.array(self.metadata['preprocessing']['normalize_std'])
        image = (image - mean) / std
        
        # Rearrange dimensions: HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        if self.format == 'torchscript':
            return torch.from_numpy(image).to(self.device)
        else:
            return image.astype(np.float32)
    
    def predict(self, image: Union[str, np.ndarray]) -> Tuple[float, str]:
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image as file path (str/Path) or numpy array.
        
        Returns:
            tuple: (probability, label) where probability is the dog confidence
                   score and label is 'DOG' or 'NOT-DOG'.
        """
        # Preprocess input
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        if self.format == 'torchscript':
            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).item()
        else:
            output = self.model.run(None, {'input': input_tensor})[0]
            probability = 1.0 / (1.0 + np.exp(-output[0]))  # sigmoid
        
        # Classify based on threshold
        is_dog = probability > 0.5
        label = "DOG" if is_dog else "NOT-DOG"
        
        return probability, label
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Tuple[float, str]]:
        """
        Make predictions on a batch of images.
        
        Args:
            images (list): List of images as file paths or numpy arrays.
        
        Returns:
            list: List of (probability, label) tuples for each image.
        """
        results = []
        
        # Preprocess all images
        batch_inputs = []
        for image in images:
            input_tensor = self.preprocess_image(image)
            if self.format == 'torchscript':
                batch_inputs.append(input_tensor)
            else:
                batch_inputs.append(input_tensor[0])  # Remove batch dimension for stacking
        
        if self.format == 'torchscript':
            # Concatenate into batch tensor
            batch_tensor = torch.cat(batch_inputs, dim=0)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
        else:
            # ONNX batch processing
            batch_array = np.stack(batch_inputs, axis=0)
            outputs = self.model.run(None, {'input': batch_array})[0]
            probabilities = 1.0 / (1.0 + np.exp(-outputs))  # sigmoid
        
        # Process results
        for prob in probabilities:
            if isinstance(prob, np.ndarray):
                prob = prob.item()
            is_dog = prob > 0.5
            label = "DOG" if is_dog else "NOT-DOG"
            results.append((prob, label))
        
        return results

if __name__ == "__main__":
    # Usage example
    model_path = "path/to/your/best_model.pth"
    
    print("Starting inference optimization...")
    
    # Create optimizer
    optimizer = InferenceOptimizer(model_path)
    
    # Optimize to TorchScript
    optimizer.optimize_to_torchscript()
    
    # Optimize to ONNX
    optimizer.optimize_to_onnx()
    
    # Run benchmarks
    results = optimizer.benchmark_models(num_runs=50)
    
    # Create production model package
    prod_model_path, metadata_path = optimizer.create_production_model('torchscript')
    
    # Test production inference
    inference = ProductionInference(prod_model_path, metadata_path)
    
    # Example prediction (requires actual image)
    # probability, label = inference.predict("path/to/test/image.jpg")
    # print(f"Result: {label} (probability: {probability:.3f})")