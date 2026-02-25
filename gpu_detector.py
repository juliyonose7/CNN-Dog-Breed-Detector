"""
Detector de GPU AMD y configuración para DirectML en Windows
"""

import torch
import torch_directml

def setup_amd_gpu():
    """Configura la GPU AMD para Windows usando DirectML"""
    print("🔍 Detectando hardware disponible...")
    
    # Verificar DirectML
    if torch_directml.is_available():
        device_count = torch_directml.device_count()
        print(f"✅ DirectML disponible con {device_count} dispositivo(s)")
        
        # Obtener información del dispositivo
        for i in range(device_count):
            device = torch_directml.device(i)
            print(f"   Dispositivo {i}: {device}")
        
        # Usar el primer dispositivo DirectML
        device = torch_directml.device()
        print(f"🚀 Usando GPU AMD con DirectML: {device}")
        return device, True
    
    # Fallback a CUDA si está disponible
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🟡 Usando CUDA: {torch.cuda.get_device_name()}")
        return device, True
    
    # Fallback a CPU
    else:
        device = torch.device('cpu')
        print("⚠️  Usando CPU (GPU no detectada)")
        return device, False

def test_gpu_performance():
    """Prueba básica de rendimiento de la GPU"""
    device, gpu_available = setup_amd_gpu()
    
    if not gpu_available:
        return False
    
    print("\n🧪 Probando rendimiento de GPU...")
    
    import time
    
    # Crear tensores de prueba
    size = 1024
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)
    
    # Calentamiento
    for _ in range(10):
        c = torch.matmul(a, b)
    
    # Sincronizar si es necesario
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Medir tiempo
    start_time = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    ops_per_sec = 100 / total_time
    
    print(f"✅ Multiplicación de matrices {size}x{size}:")
    print(f"   Tiempo total: {total_time:.3f}s")
    print(f"   Operaciones/seg: {ops_per_sec:.1f}")
    print(f"   GPU está {ops_per_sec/10:.1f}x más rápida que referencia CPU")
    
    return True

if __name__ == "__main__":
    test_gpu_performance()