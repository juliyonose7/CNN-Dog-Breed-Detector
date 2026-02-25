# configuracion centralizada del proyecto de clasificacion de razas
# este archivo contiene todas las constantes y parametros configurables
# permite modificar comportamiento sin cambiar codigo fuente

# informacion basica del proyecto
PROJECT_NAME = "Dog Classification API"  # nombre descriptivo del proyecto
VERSION = "1.0.0"                       # version actual del sistema

# definicion de rutas importantes del proyecto
# todas las rutas son relativas al directorio raiz
DATASET_PATH = "./DATASETS"              # directorio con imagenes originales
PROCESSED_DATA_PATH = "./processed_data" # directorio con datos procesados
MODELS_PATH = "./models"                 # directorio donde se guardan modelos
OPTIMIZED_MODELS_PATH = "./optimized_models"  # modelos optimizados para produccion

# configuracion del modelo de red neuronal
# define arquitectura y parametros fundamentales
MODEL_CONFIG = {
    "model_name": "efficientnet_b3",  # arquitectura: efficientnet_b3, resnet50, resnet101, densenet121
    "input_size": (224, 224),         # resolucion de imagenes de entrada en pixeles
    "num_classes": 1,                 # numero de clases a clasificar inicialmente
    "pretrained": True                # usar pesos preentrenados en imagenet
}

# configuracion de parametros de entrenamiento
# controla hiperparametros y estrategias de optimizacion
TRAINING_CONFIG = {
    "batch_size": 32,                 # numero de muestras por batch procesadas simultaneamente
    "num_epochs": 30,                 # numero total de epochs de entrenamiento
    "learning_rate": 1e-3,            # tasa de aprendizaje inicial del optimizador
    "weight_decay": 1e-4,             # regularizacion l2 para prevenir overfitting
    "freeze_epochs": 5,               # epochs con backbone congelado al inicio
    "balance_strategy": "undersample" # estrategia de balanceo: undersample, oversample, none
}

# configuracion de optimizacion para hardware amd rocm
# ajusta parametros para mejor rendimiento en gpus amd
ROCM_CONFIG = {
    "device": "cuda",              # dispositivo: cuda para rocm, cpu para fallback
    "mixed_precision": True,       # mixed precision training para mayor velocidad
    "benchmark": True,             # optimiza cudnn para mejor rendimiento
    "deterministic": False         # false permite mayor velocidad sacrificando reproducibilidad
}

# configuracion de data augmentation para mejorar generalizacion
# define transformaciones aleatorias aplicadas durante entrenamiento
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,        # probabilidad de flip horizontal
    "rotation_limit": 15,          # rotacion maxima en grados
    "brightness_contrast": 0.2,    # variacion de brillo y contraste
    "gaussian_noise": 0.3,         # ruido gaussiano para robustez
    "cutout": 0.2,                 # cutout para regularizacion
    "normalize_mean": [0.485, 0.456, 0.406],  # media de normalizacion imagenet
    "normalize_std": [0.229, 0.224, 0.225]    # desviacion estandar imagenet
}

# configuracion del servidor api para servir el modelo
# define parametros de red y limites de seguridad
API_CONFIG = {
    "host": "0.0.0.0",                      # acepta conexiones desde cualquier ip
    "port": 8000,                           # puerto donde se ejecuta el servidor
    "workers": 1,                           # numero de workers del servidor
    "max_file_size": 10 * 1024 * 1024,     # limite de 10mb por archivo
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],  # formatos permitidos
    "max_batch_size": 10                   # maximo de imagenes por batch
}

# configuracion del sistema de logging para debugging
# controla como se registran eventos y errores
LOGGING_CONFIG = {
    "level": "INFO",                        # nivel minimo de logging
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # formato de mensajes
    "log_file": "./logs/app.log"           # archivo donde se guardan los logs
}