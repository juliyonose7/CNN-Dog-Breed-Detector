# configuration centralizada of the proyecto of classification of breeds
# this file contiene all the constantes and parameters configurables
# allows modificar comportamiento without cambiar codigo fuente

# informacion basica of the proyecto
PROJECT_NAME = "Dog Classification API"  # name descriptivo of the proyecto
VERSION = "1.0.0"                       # version actual of the system

# definicion of paths importantes of the proyecto
# all the paths son relativas to the directory raiz
DATASET_PATH = "./DATASETS"              # directory with images originales
PROCESSED_DATA_PATH = "./processed_data" # directory with data procesados
MODELS_PATH = "./models"                 # directory where se guardan models
OPTIMIZED_MODELS_PATH = "./optimized_models"  # models optimizados for produccion

# configuration of the model of red neuronal
# define arquitectura and parameters fundamentales
MODEL_CONFIG = {
    "model_name": "efficientnet_b3",  # arquitectura: efficientnet_b3, resnet50, resnet101, densenet121
    "input_size": (224, 224),         # resolucion of images of input en pixeles
    "num_classes": 1,                 # number of classes a clasificar inicialmente
    "pretrained": True                # use weights preentrenados en imagenet
}

# configuration of parameters of training
# controla hiperparametros and estrategias of optimization
TRAINING_CONFIG = {
    "batch_size": 32,                 # number of samples for batch procesadas simultaneamente
    "num_epochs": 30,                 # number total of epochs of training
    "learning_rate": 1e-3,            # tasa of aprendizaje initial of the optimizador
    "weight_decay": 1e-4,             # regularizacion l2 for prevenir overfitting
    "freeze_epochs": 5,               # epochs with backbone congelado to the startup
    "balance_strategy": "undersample" # strategy of balanceo: undersample, oversample, none
}

# configuration of optimization for hardware amd rocm
# ajusta parameters for best performance en gpus amd
ROCM_CONFIG = {
    "device": "cuda",              # device: cuda for rocm, cpu for fallback
    "mixed_precision": True,       # mixed precision training for mayor velocidad
    "benchmark": True,             # optimiza cudnn for best performance
    "deterministic": False         # false allows mayor velocidad sacrificando reproducibilidad
}

# configuration of data augmentation for improve generalizacion
# define transformaciones aleatorias aplicadas durante training
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,        # probabilidad of flip horizontal
    "rotation_limit": 15,          # rotacion maxima en grados
    "brightness_contrast": 0.2,    # variacion of brillo and contraste
    "gaussian_noise": 0.3,         # ruido gaussiano for robustez
    "cutout": 0.2,                 # cutout for regularizacion
    "normalize_mean": [0.485, 0.456, 0.406],  # media of normalizacion imagenet
    "normalize_std": [0.229, 0.224, 0.225]    # deviation estandar imagenet
}

# configuration of the server API for servir the model
# define parameters of red and limites of seguridad
API_CONFIG = {
    "host": "0.0.0.0",                      # acepta conexiones from cualquier ip
    "port": 8000,                           # port where se ejecuta the server
    "workers": 1,                           # number of workers of the server
    "max_file_size": 10 * 1024 * 1024,     # limite of 10mb for file
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],  # formatos permitidos
    "max_batch_size": 10                   # maximo of images for batch
}

# configuration of the system of logging for debugging
# controla como se registran eventos and errors
LOGGING_CONFIG = {
    "level": "INFO",                        # nivel minimo of logging
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # format of mensajes
    "log_file": "./logs/app.log"           # file where se guardan the logs
}