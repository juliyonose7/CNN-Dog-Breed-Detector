#!/usr/bin/env python3
"""
Comprehensive Technical Documentation Generator.
This script automatically adds or improves technical English comments
across all Python files in the project. It handles:
- Module-level docstrings
- Class docstrings
- Function/method docstrings
- Inline comments translation
- Code section headers
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# =============================================================================
# TRANSLATION DICTIONARY (Spanish -> Technical English)
# =============================================================================
SPANISH_TO_ENGLISH = {
    # Common technical terms
    "configurar": "configure",
    "configuración": "configuration",
    "inicializar": "initialize",
    "obtener": "get",
    "cargar": "load",
    "guardar": "save",
    "crear": "create",
    "eliminar": "delete",
    "actualizar": "update",
    "validar": "validate",
    "procesar": "process",
    "ejecutar": "execute",
    "entrenar": "train",
    "predecir": "predict",
    "clasificar": "classify",
    "evaluar": "evaluate",
    "optimizar": "optimize",
    "analizar": "analyze",
    "generar": "generate",
    "calcular": "calculate",
    "mostrar": "display",
    "imprimir": "print",
    "devolver": "return",
    "verificar": "verify",
    "comprobar": "check",
    "detectar": "detect",
    "extraer": "extract",
    "transformar": "transform",
    "normalizar": "normalize",
    "escalar": "scale",
    "ajustar": "adjust",
    "modificar": "modify",
    "establecer": "set",
    "definir": "define",
    "determinar": "determine",
    "iniciar": "start",
    "finalizar": "finish",
    "terminar": "terminate",
    "pausar": "pause",
    "reanudar": "resume",
    "cancelar": "cancel",
    "reiniciar": "restart",
    "limpiar": "clean",
    "filtrar": "filter",
    "ordenar": "sort",
    "buscar": "search",
    "encontrar": "find",
    "seleccionar": "select",
    "elegir": "choose",
    "añadir": "add",
    "agregar": "add",
    "insertar": "insert",
    "remover": "remove",
    "quitar": "remove",
    "mezclar": "shuffle",
    "barajar": "shuffle",
    "dividir": "split",
    "combinar": "combine",
    "unir": "merge",
    "concatenar": "concatenate",
    "duplicar": "duplicate",
    "copiar": "copy",
    "mover": "move",
    "enviar": "send",
    "recibir": "receive",
    "conectar": "connect",
    "desconectar": "disconnect",
    "sincronizar": "synchronize",
    "exportar": "export",
    "importar": "import",
    "convertir": "convert",
    "parsear": "parse",
    "serializar": "serialize",
    "deserializar": "deserialize",
    "codificar": "encode",
    "decodificar": "decode",
    "comprimir": "compress",
    "descomprimir": "decompress",
    "cifrar": "encrypt",
    "descifrar": "decrypt",
    
    # ML/DL specific terms
    "modelo": "model",
    "modelos": "models",
    "entrenamiento": "training",
    "validación": "validation",
    "prueba": "test",
    "época": "epoch",
    "épocas": "epochs",
    "lote": "batch",
    "lotes": "batches",
    "pérdida": "loss",
    "precisión": "accuracy",
    "exactitud": "accuracy",
    "predicción": "prediction",
    "predicciones": "predictions",
    "inferencia": "inference",
    "clasificación": "classification",
    "regresión": "regression",
    "optimizador": "optimizer",
    "gradiente": "gradient",
    "gradientes": "gradients",
    "pesos": "weights",
    "sesgos": "biases",
    "parámetros": "parameters",
    "hiperparámetros": "hyperparameters",
    "capas": "layers",
    "capa": "layer",
    "neurona": "neuron",
    "neuronas": "neurons",
    "activación": "activation",
    "normalización": "normalization",
    "regularización": "regularization",
    "dropout": "dropout",
    "overfitting": "overfitting",
    "underfitting": "underfitting",
    "generalización": "generalization",
    "aumentación": "augmentation",
    "aumento": "augmentation",
    "transformación": "transformation",
    "preprocesamiento": "preprocessing",
    "postprocesamiento": "postprocessing",
    "extracción": "extraction",
    "características": "features",
    "etiqueta": "label",
    "etiquetas": "labels",
    "clase": "class",
    "clases": "classes",
    "categoría": "category",
    "categorías": "categories",
    "muestra": "sample",
    "muestras": "samples",
    "conjunto": "set",
    "dataset": "dataset",
    "tensor": "tensor",
    "tensores": "tensors",
    "matriz": "matrix",
    "matrices": "matrices",
    "vector": "vector",
    "vectores": "vectors",
    "dimensión": "dimension",
    "dimensiones": "dimensions",
    "forma": "shape",
    "tamaño": "size",
    "resolución": "resolution",
    "imagen": "image",
    "imágenes": "images",
    "raza": "breed",
    "razas": "breeds",
    "perro": "dog",
    "perros": "dogs",
    "canino": "canine",
    "mascota": "pet",
    "animal": "animal",
    
    # System/API terms
    "servidor": "server",
    "cliente": "client",
    "solicitud": "request",
    "respuesta": "response",
    "endpoint": "endpoint",
    "ruta": "route",
    "puerto": "port",
    "host": "host",
    "conexión": "connection",
    "sesión": "session",
    "autenticación": "authentication",
    "autorización": "authorization",
    "error": "error",
    "excepción": "exception",
    "advertencia": "warning",
    "información": "information",
    "mensaje": "message",
    "registro": "log",
    "archivo": "file",
    "archivos": "files",
    "directorio": "directory",
    "carpeta": "folder",
    "ruta": "path",
    "rutas": "paths",
    "configuración": "configuration",
    "ajustes": "settings",
    "opciones": "options",
    "parámetro": "parameter",
    "argumento": "argument",
    "valor": "value",
    "valores": "values",
    "resultado": "result",
    "resultados": "results",
    "salida": "output",
    "entrada": "input",
    "estado": "state",
    "status": "status",
    "tiempo": "time",
    "fecha": "date",
    "duración": "duration",
    "memoria": "memory",
    "caché": "cache",
    "almacenamiento": "storage",
    "dispositivo": "device",
    "hardware": "hardware",
    "software": "software",
    "versión": "version",
    "número": "number",
    "cantidad": "quantity",
    "total": "total",
    "máximo": "maximum",
    "mínimo": "minimum",
    "promedio": "average",
    "media": "mean",
    "desviación": "deviation",
    "varianza": "variance",
    "umbral": "threshold",
    "límite": "limit",
    "rango": "range",
    "intervalo": "interval",
    
    # Additional technical Spanish
    "disponible": "available",
    "activo": "active",
    "inactivo": "inactive",
    "habilitado": "enabled",
    "deshabilitado": "disabled",
    "completo": "complete",
    "incompleto": "incomplete",
    "válido": "valid",
    "inválido": "invalid",
    "exitoso": "successful",
    "fallido": "failed",
    "pendiente": "pending",
    "procesando": "processing",
    "finalizado": "finished",
    "terminado": "completed",
    "cancelado": "cancelled",
    "requerido": "required",
    "opcional": "optional",
    "obligatorio": "mandatory",
    "predeterminado": "default",
    "personalizado": "custom",
    "público": "public",
    "privado": "private",
    "protegido": "protected",
    "interno": "internal",
    "externo": "external",
    "local": "local",
    "remoto": "remote",
    "temporal": "temporary",
    "permanente": "permanent",
    "global": "global",
    "estático": "static",
    "dinámico": "dynamic",
    "abstracto": "abstract",
    "concreto": "concrete",
    "genérico": "generic",
    "específico": "specific",
    "principal": "main",
    "secundario": "secondary",
    "auxiliar": "auxiliary",
    "adicional": "additional",
    "anterior": "previous",
    "siguiente": "next",
    "actual": "current",
    "nuevo": "new",
    "antiguo": "old",
    "mejor": "best",
    "peor": "worst",
    "primero": "first",
    "último": "last",
    "único": "unique",
    "múltiple": "multiple",
    "simple": "simple",
    "complejo": "complex",
    "básico": "basic",
    "avanzado": "advanced",
    "inicial": "initial",
    "final": "final",
}

# =============================================================================
# MODULE DOCSTRING TEMPLATES
# =============================================================================
MODULE_TEMPLATES = {
    "api": '''"""
{name} - REST API Module for Dog Classification System.
Provides HTTP endpoints for image classification and inference services.
Supports single image and batch processing with JSON responses.
"""''',
    
    "trainer": '''"""
{name} - Neural Network Training Module.
Implements training loops, optimization strategies, and model checkpointing
for deep learning classifier models with support for AMD GPU acceleration.
"""''',
    
    "classifier": '''"""
{name} - Image Classification Module.
Contains neural network architectures and inference logic for classifying
images using pretrained models with transfer learning.
"""''',
    
    "dataset": '''"""
{name} - Dataset Management Module.
Handles data loading, preprocessing, and augmentation for training
and validation image datasets with class balancing support.
"""''',
    
    "analyzer": '''"""
{name} - Data Analysis Module.
Provides tools for analyzing model performance, dataset statistics,
and generating evaluation metrics and visualizations.
"""''',
    
    "config": '''"""
{name} - Configuration Module.
Defines constants, hyperparameters, and configurable settings for
the classification system with centralized parameter management.
"""''',
    
    "utils": '''"""
{name} - Utility Functions Module.
Contains helper functions and common utilities used across
the classification system for data manipulation and I/O operations.
"""''',
    
    "default": '''"""
{name} - Support Module.
Provides additional functionality for the dog breed classification system.
"""'''
}


def get_module_type(filename: str, content: str) -> str:
    """
    Determine the module type based on filename and content analysis.
    
    Args:
        filename: Name of the Python file.
        content: File content for keyword analysis.
    
    Returns:
        str: Module type key for template selection.
    """
    name_lower = filename.lower()
    content_lower = content.lower()
    
    # Priority order for type detection
    if "api" in name_lower or "server" in name_lower or "fastapi" in content_lower:
        return "api"
    elif "trainer" in name_lower or "training" in name_lower:
        return "trainer"
    elif "classifier" in name_lower or "model" in content_lower[:500]:
        return "classifier"
    elif "dataset" in name_lower or "dataloader" in content_lower:
        return "dataset"
    elif "analyze" in name_lower or "analyzer" in name_lower or "analysis" in name_lower:
        return "analyzer"
    elif "config" in name_lower:
        return "config"
    elif "utils" in name_lower or "helper" in name_lower:
        return "utils"
    else:
        return "default"


def translate_text(text: str) -> str:
    """
    Translate Spanish text to technical English using the dictionary.
    
    Args:
        text: Input text potentially containing Spanish words.
    
    Returns:
        str: Text with Spanish words replaced by English equivalents.
    """
    result = text
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(SPANISH_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True)
    
    for spanish, english in sorted_terms:
        # Case-insensitive replacement while preserving case
        pattern = re.compile(re.escape(spanish), re.IGNORECASE)
        
        def replace_match(match):
            original = match.group()
            if original.isupper():
                return english.upper()
            elif original[0].isupper():
                return english.capitalize()
            else:
                return english
        
        result = pattern.sub(replace_match, result)
    
    return result


def needs_documentation(content: str) -> bool:
    """
    Check if a file needs documentation improvements.
    
    Args:
        content: File content to analyze.
    
    Returns:
        bool: True if documentation is missing or poor quality.
    """
    # Check for Spanish characters or patterns
    spanish_patterns = [
        r'[áéíóúñü]',  # Accented characters
        r'\b(del|al|el|la|los|las|una|uno|con|para|que|por|como)\b',  # Spanish articles/prepositions
        r'\b(este|esta|estos|estas|ese|esa|esos|esas)\b',  # Demonstratives
        r'\b(muy|más|menos|mejor|peor|también|además)\b',  # Adverbs
    ]
    
    for pattern in spanish_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    
    # Check for minimal docstrings
    if content.count('"""') < 4:
        return True
    
    return False


def add_section_comments(content: str) -> str:
    """
    Add section header comments to organize code blocks.
    
    Args:
        content: Original file content.
    
    Returns:
        str: Content with section headers added.
    """
    lines = content.split('\n')
    result_lines = []
    
    import_section_started = False
    class_section_started = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect import section start
        if stripped.startswith('import ') or stripped.startswith('from '):
            if not import_section_started:
                import_section_started = True
                # Check if previous line is already a comment
                if i > 0 and not lines[i-1].strip().startswith('#'):
                    result_lines.append('')
                    result_lines.append('# =============================================================================')
                    result_lines.append('# IMPORTS')
                    result_lines.append('# =============================================================================')
        
        # Detect class definition
        if stripped.startswith('class ') and not class_section_started:
            class_section_started = True
            if i > 0 and not lines[i-1].strip().startswith('#') and not lines[i-1].strip() == '':
                result_lines.append('')
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def process_file(filepath: Path) -> Tuple[bool, str]:
    """
    Process a single Python file to add/improve documentation.
    
    Args:
        filepath: Path to the Python file.
    
    Returns:
        Tuple[bool, str]: (was_modified, status_message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not needs_documentation(content):
            return False, "Already documented"
        
        original_content = content
        
        # Step 1: Translate Spanish comments and docstrings
        content = translate_text(content)
        
        # Step 2: Check/add module docstring
        filename = filepath.stem
        module_type = get_module_type(filename, content)
        template = MODULE_TEMPLATES[module_type].format(name=filename)
        
        # Check if module docstring exists and is adequate
        docstring_pattern = r'^(\s*["\'][\'"]{2}.*?["\'][\'"]{2}\s*)'
        match = re.match(docstring_pattern, content, re.DOTALL)
        
        if not match:
            # No module docstring - add one
            content = template + '\n\n' + content
        elif len(match.group(1)) < 50:
            # Replace minimal docstring
            content = re.sub(docstring_pattern, template + '\n\n', content, count=1, flags=re.DOTALL)
        
        # Step 3: Add section comments
        content = add_section_comments(content)
        
        # Check if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Updated"
        
        return False, "No changes needed"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def process_directory(directory: Path, exclude_dirs: List[str] = None) -> Dict[str, int]:
    """
    Process all Python files in a directory.
    
    Args:
        directory: Root directory to process.
        exclude_dirs: List of directory names to skip.
    
    Returns:
        Dict[str, int]: Statistics about processed files.
    """
    if exclude_dirs is None:
        exclude_dirs = ['.venv', 'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']
    
    stats = {
        'total': 0,
        'updated': 0,
        'skipped': 0,
        'errors': 0
    }
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE DOCUMENTATION GENERATOR")
    print(f"{'='*70}")
    print(f"Processing directory: {directory}\n")
    
    for filepath in directory.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in filepath.parts for excluded in exclude_dirs):
            continue
        
        stats['total'] += 1
        modified, status = process_file(filepath)
        
        rel_path = filepath.relative_to(directory)
        
        if modified:
            stats['updated'] += 1
            print(f"[OK] {rel_path}: {status}")
        elif "Error" in status:
            stats['errors'] += 1
            print(f"[ERROR] {rel_path}: {status}")
        else:
            stats['skipped'] += 1
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total files:   {stats['total']}")
    print(f"Updated:       {stats['updated']}")
    print(f"Skipped:       {stats['skipped']}")
    print(f"Errors:        {stats['errors']}")
    print(f"{'='*70}\n")
    
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path(__file__).parent.parent
    
    process_directory(target_dir)
