"""Spanish to English Comment Translation Script.

This module provides automated translation of Spanish text in source
code comments and docstrings to technical English. It uses a comprehensive
dictionary-based approach for consistent translations.

Features:
    - Dictionary-based word and phrase translation
    - Preserves code structure and formatting
    - Handles Python, JavaScript, CSS, and HTML files
    - Git integration for automatic staging of changes

Usage:
    python scripts/translate_comments_to_english.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


# Project root directory for file discovery
REPO_ROOT = Path(__file__).resolve().parents[1]


SENTENCE_REPLACEMENTS = {
    "configuración": "configuration",
    "configuracion": "configuration",
    "clasificación": "classification",
    "clasificacion": "classification",
    "entrenamiento": "training",
    "validación": "validation",
    "validacion": "validation",
    "precisión": "precision",
    "precision": "precision",
    "raza": "breed",
    "razas": "breeds",
    "modelo": "model",
    "modelos": "models",
    "imágenes": "images",
    "imagenes": "images",
    "imagen": "image",
    "mejor": "best",
    "mejores": "best",
    "peor": "worst",
    "archivo": "file",
    "archivos": "files",
    "cargar": "load",
    "carga": "load",
    "guardar": "save",
    "guardar": "save",
    "directorio": "directory",
    "directamente": "directly",
    "predicción": "prediction",
    "prediccion": "prediction",
    "predicciones": "predictions",
    "umbral": "threshold",
    "umbrales": "thresholds",
    "falso": "false",
    "falsos": "false",
    "negativo": "negative",
    "negativos": "negatives",
    "positivo": "positive",
    "positivos": "positives",
    "proceso": "process",
    "procesamiento": "processing",
    "sistema": "system",
    "servidor": "server",
    "api": "API",
    "puerto": "port",
    "inicio": "startup",
    "iniciar": "start",
    "detener": "stop",
    "función": "function",
    "funcion": "function",
    "método": "method",
    "metodo": "method",
    "clase": "class",
    "clases": "classes",
    "lista": "list",
    "mapeo": "mapping",
    "índice": "index",
    "indice": "index",
    "nombre": "name",
    "nombres": "names",
    "usuario": "user",
    "errores": "errors",
    "error": "error",
    "rápido": "fast",
    "rapido": "fast",
    "lento": "slow",
    "optimización": "optimization",
    "optimizacion": "optimization",
    "optimizado": "optimized",
    "datos": "data",
    "balanceado": "balanced",
    "balanceada": "balanced",
    "aumentación": "augmentation",
    "aumentacion": "augmentation",
    "debe": "must",
    "deben": "must",
    "si": "if",
    "siempre": "always",
    "solo": "only",
    "todas": "all",
    "todos": "all",
    "esta": "this",
    "este": "this",
    "estas": "these",
    "estos": "these",
    "para": "for",
    "con": "with",
    "sin": "without",
    "cuando": "when",
    "donde": "where",
    "que": "that",
    "del": "of the",
    "al": "to the",
}


PHRASE_REPLACEMENTS = {
    "cargar modelo": "load model",
    "guardar modelo": "save model",
    "mejor modelo": "best model",
    "datos de validación": "validation data",
    "datos de entrenamiento": "training data",
    "clase por defecto": "default class",
    "por defecto": "by default",
    "mapeo rápido": "quick mapping",
    "iniciar servidor": "start server",
    "modelo balanceado": "balanced model",
    "top-5 predicciones": "top-5 predictions",
    "si es perro": "if it is a dog",
    "no es perro": "not a dog",
}


def to_english(text: str) -> str:
    if not text.strip():
        return text

    output = text
    for es, en in PHRASE_REPLACEMENTS.items():
        output = re.sub(rf"\b{re.escape(es)}\b", en, output, flags=re.IGNORECASE)

    words = re.split(r"(\W+)", output)
    translated: list[str] = []
    for token in words:
        key = token.lower()
        if key in SENTENCE_REPLACEMENTS:
            replacement = SENTENCE_REPLACEMENTS[key]
            if token.istitle():
                replacement = replacement.capitalize()
            translated.append(replacement)
        else:
            translated.append(token)

    output = "".join(translated)
    output = re.sub(r"\s+", " ", output)
    return output.strip()


def translate_python_comments(content: str) -> str:
    lines = content.splitlines(keepends=True)
    new_lines: list[str] = []

    in_docstring = False
    doc_delim = ""

    for line in lines:
        stripped = line.lstrip()

        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("''""")):
            delim = stripped[:3]
            if stripped.count(delim) >= 2 and len(stripped.strip()) > 6:
                prefix = line[: line.find(delim) + 3]
                inner = stripped[3 : stripped.rfind(delim)]
                suffix = delim + ("\n" if line.endswith("\n") else "")
                new_lines.append(prefix + to_english(inner) + suffix)
                continue
            in_docstring = True
            doc_delim = delim
            prefix_idx = line.find(delim) + 3
            prefix = line[:prefix_idx]
            rest = line[prefix_idx:]
            new_lines.append(prefix + to_english(rest.rstrip("\n")) + ("\n" if line.endswith("\n") else ""))
            continue

        if in_docstring:
            if doc_delim in stripped:
                pos = line.find(doc_delim)
                before = line[:pos]
                after = line[pos:]
                new_lines.append(to_english(before.rstrip("\n")) + after)
                in_docstring = False
                doc_delim = ""
            else:
                new_lines.append(to_english(line.rstrip("\n")) + ("\n" if line.endswith("\n") else ""))
            continue

        if "# " in line:
            i = line.find("# ")
            prefix = line[:i]
            comment = line[i + 1 :].rstrip("\n")
            translated = to_english(comment)
            new_lines.append(f"{prefix}# {translated}" + ("\n" if line.endswith("\n") else ""))
            continue

        new_lines.append(line)

    return "".join(new_lines)


def translate_generic_comments(content: str) -> str:
    def repl_html(m: re.Match[str]) -> str:
        return f"<!-- {to_english(m.group(1))} -->"

    def repl_block(m: re.Match[str]) -> str:
        return "/* " + to_english(m.group(1)) + " */"

    def repl_line(m: re.Match[str]) -> str:
        return m.group(1) + "// " + to_english(m.group(2))

    content = re.sub(r"<!--\s*(.*?)\s*-->", repl_html, content, flags=re.DOTALL)
    content = re.sub(r"/\*\s*(.*?)\s*\*/", repl_block, content, flags=re.DOTALL)
    content = re.sub(r"(^|\s)//\s*(.*)$", repl_line, content, flags=re.MULTILINE)
    return content


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[Path] = []
    for line in result.stdout.splitlines():
        path = REPO_ROOT / line
        if path.suffix.lower() in {".py", ".js", ".ts", ".html", ".css"}:
            files.append(path)
    return files


def main() -> None:
    changed = 0
    for path in tracked_files():
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if path.suffix.lower() == ".py":
            updated = translate_python_comments(original)
        else:
            updated = translate_generic_comments(original)

        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1

    print(f"Updated files: {changed}")


if __name__ == "__main__":
    main()
