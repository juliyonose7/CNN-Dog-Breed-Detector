from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


SPANISH_HINTS = re.compile(
    r"[áéíóúñÁÉÍÓÚÑ]|\b(que|para|con|sin|solo|si|esta|este|estas|estos|cuando|donde|porque|raza|razas|modelo|modelos|imagen|imagenes|prediccion|predicción|umbral|umbrales|entrenamiento|validacion|validación|métrica|métricas|cargar|guardar|mejorar|clasificacion|clasificación|dispositivo|servidor|archivo|archivos)\b",
    re.IGNORECASE,
)


PHRASES = {
    "reentrenamiento selectivo": "selective retraining",
    "modelo preentrenado": "pretrained model",
    "mejor modelo": "best model",
    "datos de validación": "validation data",
    "datos de entrenamiento": "training data",
    "si es perro": "if a dog is detected",
    "clasificación de raza": "breed classification",
    "solo detección": "detection only",
    "clases": "classes",
    "clase": "class",
}


WORDS = {
    "de": "of",
    "del": "of the",
    "la": "the",
    "el": "the",
    "las": "the",
    "los": "the",
    "y": "and",
    "o": "or",
    "un": "a",
    "una": "a",
    "unos": "some",
    "unas": "some",
    "por": "for",
    "desde": "from",
    "hasta": "until",
    "sobre": "about",
    "hacia": "toward",
    "mas": "more",
    "más": "more",
    "menos": "less",
    "muy": "very",
    "algunas": "some",
    "algunos": "some",
    "all": "all",
    "allas": "all",
    "allos": "all",
    "obtener": "get",
    "obtiene": "gets",
    "crea": "creates",
    "creado": "created",
    "creada": "created",
    "mostrar": "show",
    "muestra": "shows",
    "verificar": "verify",
    "verifica": "verifies",
    "buscar": "search",
    "busca": "searches",
    "usar": "use",
    "usa": "uses",
    "hacer": "do",
    "hecho": "done",
    "aplicar": "apply",
    "aplica": "applies",
    "coincida": "matches",
    "coincidir": "match",
    "evita": "avoids",
    "detenga": "stops",
    "detener": "stop",
    "corresponden": "match",
    "corresponde": "matches",
    "existe": "exists",
    "existan": "exist",
    "valida": "valid",
    "válida": "valid",
    "validos": "valid",
    "válidos": "valid",
    "criticos": "critical",
    "críticos": "critical",
    "rapido": "fast",
    "rápido": "fast",
    "lenta": "slow",
    "mejor": "best",
    "peor": "worst",
    "nuevo": "new",
    "nueva": "new",
    "esperar": "wait",
    "espera": "wait",
    "decision": "decision",
    "decida": "decides",
    "tiempo": "time",
    "barra": "bar",
    "progreso": "progress",
    "etapa": "stage",
    "estado": "status",
    "fondo": "background",
    "principal": "main",
    "globales": "global",
    "global": "global",
    "completa": "complete",
    "completo": "complete",
    "detalladas": "detailed",
    "detallada": "detailed",
    "detallado": "detailed",
    "mejora": "improvement",
    "balanceado": "balanced",
    "balanceada": "balanced",
    "salida": "output",
    "entrada": "input",
    "entorno": "environment",
    "capa": "layer",
    "capas": "layers",
    "pesos": "weights",
    "final": "final",
    "inicial": "initial",
    "prueba": "test",
    "pruebas": "tests",
    "graficos": "plots",
    "gráficos": "plots",
    "grafica": "plot",
    "gráfica": "plot",
    "distribucion": "distribution",
    "distribución": "distribution",
    "comparacion": "comparison",
    "comparación": "comparison",
    "estimacion": "estimate",
    "estimación": "estimate",
    "promedio": "average",
    "desviacion": "deviation",
    "desviación": "deviation",
    "actualizar": "update",
    "actualiza": "updates",
    "reiniciar": "reset",
    "reinicia": "resets",
    "subir": "upload",
    "seleccion": "selection",
    "selección": "selection",
    "arrastrar": "drag",
    "soltar": "drop",
    "oculta": "hidden",
    "mostrar": "show",
    "muestra": "shows",
    "cambia": "changes",
    "quita": "removes",
    "agrega": "adds",
    "agregar": "add",
    "lista": "list",
    "nombre": "name",
    "nombres": "names",
    "formato": "format",
    "legible": "readable",
    "humanos": "humans",
    "framework": "framework",
    "puede": "can",
    "pueden": "can",
    "debe": "must",
    "deben": "must",
    "sirve": "serves",
    "sirvan": "serve",
    "zona": "zone",
    "carpeta": "folder",
    "carpetas": "folders",
    "subdirectorio": "subdirectory",
    "subdirectorios": "subdirectories",
    "metodo": "method",
    "método": "method",
    "metodos": "methods",
    "métodos": "methods",
    "funcion": "function",
    "función": "function",
    "control": "control",
    "detectar": "detect",
    "deteccion": "detection",
    "detección": "detection",
    "perro": "dog",
    "perros": "dogs",
    "web": "web",
    "reentrenamiento": "retraining",
    "selectivo": "selective",
    "problematicas": "problematic",
    "problemáticas": "problematic",
    "permite": "allows",
    "ciertas": "specific",
    "muestran": "show",
    "baja": "low",
    "precision": "precision",
    "precisión": "precision",
    "mejorar": "improve",
    "rendimiento": "performance",
    "afectar": "affect",
    "resto": "rest",
    "sistema": "system",
    "operativo": "operating",
    "manejo": "handling",
    "archivos": "files",
    "archivo": "file",
    "operaciones": "operations",
    "rutas": "paths",
    "numeros": "numbers",
    "aleatorios": "random",
    "entrenamiento": "training",
    "validación": "validation",
    "validacion": "validation",
    "imágenes": "images",
    "imagenes": "images",
    "imagen": "image",
    "raza": "breed",
    "razas": "breeds",
    "modelo": "model",
    "modelos": "models",
    "cargar": "load",
    "carga": "load",
    "guardar": "save",
    "guardado": "saved",
    "directorio": "directory",
    "directorios": "directories",
    "objetivo": "target",
    "objetivo": "target",
    "especificas": "specific",
    "específicas": "specific",
    "seleccionadas": "selected",
    "inicializa": "initializes",
    "parametros": "parameters",
    "parámetros": "parameters",
    "numero": "number",
    "número": "number",
    "muestras": "samples",
    "etiqueta": "label",
    "etiquetas": "labels",
    "devolver": "return",
    "devuelve": "returns",
    "estrategia": "strategy",
    "recuperacion": "recovery",
    "recuperación": "recovery",
    "corruptas": "corrupted",
    "crear": "create",
    "creando": "creating",
    "mover": "move",
    "aleatoriamente": "randomly",
    "dispositivo": "device",
    "iniciando": "starting",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for rel in result.stdout.splitlines():
        path = REPO_ROOT / rel
        if path.suffix.lower() in {".py", ".js", ".ts", ".html", ".css"}:
            files.append(path)
    return files


def is_spanish_like(text: str) -> bool:
    return bool(SPANISH_HINTS.search(text))


def translate_text(text: str) -> str:
    if not text.strip():
        return text

    output = text
    for src, dst in PHRASES.items():
        output = re.sub(rf"\b{re.escape(src)}\b", dst, output, flags=re.IGNORECASE)

    parts = re.split(r"(\W+)", output)
    translated: list[str] = []
    for token in parts:
        key = token.lower()
        if key in WORDS:
            value = WORDS[key]
            if token.istitle():
                value = value.capitalize()
            translated.append(value)
        else:
            translated.append(token)

    output = "".join(translated)
    output = re.sub(r"\s+", " ", output).strip()

    if is_spanish_like(output):
        return "Technical implementation note."
    return output


def normalize_python(content: str) -> str:
    lines = content.splitlines(keepends=True)
    new_lines: list[str] = []
    in_docstring = False
    delimiter = ""

    for line in lines:
        stripped = line.lstrip()

        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            delimiter = stripped[:3]
            if stripped.count(delimiter) >= 2 and len(stripped.strip()) > 6:
                start = line.find(delimiter)
                end = line.rfind(delimiter)
                prefix = line[: start + 3]
                inner = line[start + 3 : end]
                suffix = line[end:]
                new_lines.append(prefix + translate_text(inner) + suffix)
                continue
            in_docstring = True
            idx = line.find(delimiter) + 3
            prefix = line[:idx]
            body = line[idx:].rstrip("\n")
            new_lines.append(prefix + translate_text(body) + ("\n" if line.endswith("\n") else ""))
            continue

        if in_docstring:
            if delimiter in stripped:
                pos = line.find(delimiter)
                before = line[:pos]
                after = line[pos:]
                new_lines.append(translate_text(before.rstrip("\n")) + after)
                in_docstring = False
                delimiter = ""
            else:
                body = line.rstrip("\n")
                new_lines.append(translate_text(body) + ("\n" if line.endswith("\n") else ""))
            continue

        if "# " in line:
            idx = line.find("# ")
            prefix = line[:idx]
            comment = line[idx + 1 :].rstrip("\n")
            translated = translate_text(comment)
            new_lines.append(f"{prefix}# {translated}" + ("\n" if line.endswith("\n") else ""))
            continue

        new_lines.append(line)

    return "".join(new_lines)


def normalize_web(content: str) -> str:
    def html_repl(m: re.Match[str]) -> str:
        return f"<!-- {translate_text(m.group(1))} -->"

    def block_repl(m: re.Match[str]) -> str:
        return f"/* {translate_text(m.group(1))} */"

    def line_repl(m: re.Match[str]) -> str:
        return m.group(1) + "// " + translate_text(m.group(2))

    updated = re.sub(r"<!--\s*(.*?)\s*-->", html_repl, content, flags=re.DOTALL)
    updated = re.sub(r"/\*\s*(.*?)\s*\*/", block_repl, updated, flags=re.DOTALL)
    updated = re.sub(r"(^|\s)//\s*(.*)$", line_repl, updated, flags=re.MULTILINE)
    return updated


def main() -> None:
    updated_count = 0
    for path in tracked_files():
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if path.suffix.lower() == ".py":
            updated = normalize_python(original)
        else:
            updated = normalize_web(original)

        if updated != original:
            path.write_text(updated, encoding="utf-8")
            updated_count += 1

    print(f"Updated files: {updated_count}")


if __name__ == "__main__":
    main()
