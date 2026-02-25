# configuration centralizada of the proyecto de classification de breeds
# this file contiene all las constantes y parametros configurables
# permite modificar comportamiento without cambiar codigo fuente

# informacion basica of the proyecto
PROJECT_NAME = "Dog Classification API"  # name descriptivo of the proyecto
VERSION = "1.0.0"                       # version actual of the system

# definicion de rutas importantes of the proyecto
# all las rutas son relativas to the directory raiz
DATASET_PATH = "./DATASETS"              # directory with images originales
PROCESSED_DATA_PATH = "./processed_data" # directory with data procesados
MODELS_PATH = "./models"                 # directory where se guardan models
OPTIMIZED_MODELS_PATH = "./optimized_models"  # models optimizados for produccion

# configuration of the model de red neuronal
# define arquitectura y parametros fundamentales
MODEL_CONFIG = {
    "model_name": "efficientnet_b3",  # arquitectura: efficientnet_b3, resnet50, resnet101, densenet121
    "input_size": (224, 224),         # resolucion de images de entrada en pixeles
    "num_classes": 1,                 # numero de classes a clasificar inicialmente
    "pretrained": True                # usar pesos preentrenados en imagenet
}

# configuration de parametros de training
# controla hiperparametros y estrategias de optimization
TRAINING_CONFIG = {
    "batch_size": 32,                 # numero de muestras por batch procesadas simultaneamente
    "num_epochs": 30,                 # numero total de epochs de training
    "learning_rate": 1e-3,            # tasa de aprendizaje inicial of the optimizador
    "weight_decay": 1e-4,             # regularizacion l2 for prevenir overfitting
    "freeze_epochs": 5,               # epochs with backbone congelado to the startup
    "balance_strategy": "undersample" # estrategia de balanceo: undersample, oversample, none
}

# configuration de optimization for hardware amd rocm
# ajusta parametros for best rendimiento en gpus amd
ROCM_CONFIG = {
    "device": "cuda",              # dispositivo: cuda for rocm, cpu for fallback
    "mixed_precision": True,       # mixed precision training for mayor velocidad
    "benchmark": True,             # optimiza cudnn for best rendimiento
    "deterministic": False         # false permite mayor velocidad sacrificando reproducibilidad
}

# configuration de data augmentation for mejorar generalizacion
# define transformaciones aleatorias aplicadas durante training
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,        # probabilidad de flip horizontal
    "rotation_limit": 15,          # rotacion maxima en grados
    "brightness_contrast": 0.2,    # variacion de brillo y contraste
    "gaussian_noise": 0.3,         # ruido gaussiano for robustez
    "cutout": 0.2,                 # cutout for regularizacion
    "normalize_mean": [0.485, 0.456, 0.406],  # media de normalizacion imagenet
    "normalize_std": [0.229, 0.224, 0.225]    # desviacion estandar imagenet
}

# configuration of the server API for servir el model
# define parametros de red y limites de seguridad
API_CONFIG = {
    "host": "0.0.0.0",                      # acepta conexiones desde cualquier ip
    "port": 8000,                           # port where se ejecuta el server
    "workers": 1,                           # numero de workers of the server
    "max_file_size": 10 * 1024 * 1024,     # limite de 10mb por file
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],  # formatos permitidos
    "max_batch_size": 10                   # maximo de images por batch
}

# configuration of the system de logging for debugging
# controla como se registran eventos y errors
LOGGING_CONFIG = {
    "level": "INFO",                        # nivel minimo de logging
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # formato de mensajes
    "log_file": "./logs/app.log"           # file where se guardan los logs
}