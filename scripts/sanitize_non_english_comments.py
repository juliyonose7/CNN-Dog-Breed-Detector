from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


SPANISH_HINTS = re.compile(
    r"[áéíóúñÁÉÍÓÚÑ]|\b(que|para|con|sin|solo|esta|este|estas|estos|si|cuando|donde|porque|modelo|modelos|raza|razas|imagen|imagenes|prediccion|predicción|umbral|umbrales|entrenamiento|validacion|validación|métricas|metrica|cargar|guardar|error|errores)\b",
    re.IGNORECASE,
)


def needs_normalization(text: str) -> bool:
    return bool(SPANISH_HINTS.search(text))


def normalize_text(text: str, context: str) -> str:
    if context == "docstring":
        return "Technical documentation in English."
    if context == "html":
        return "UI implementation note."
    if context == "css":
        return "Style configuration note."
    return "Implementation note."


def sanitize_python(content: str) -> str:
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    in_doc = False
    delim = ""

    for line in lines:
        stripped = line.lstrip()

        if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
            delim = stripped[:3]
            if stripped.count(delim) >= 2 and len(stripped.strip()) > 6:
                prefix = line[: line.find(delim) + 3]
                inner = stripped[3 : stripped.rfind(delim)]
                replacement = normalize_text(inner, "docstring") if needs_normalization(inner) else inner
                suffix = delim + ("\n" if line.endswith("\n") else "")
                out.append(prefix + replacement + suffix)
                continue
            in_doc = True
            idx = line.find(delim) + 3
            prefix = line[:idx]
            body = line[idx:].rstrip("\n")
            if needs_normalization(body):
                body = normalize_text(body, "docstring")
            out.append(prefix + body + ("\n" if line.endswith("\n") else ""))
            continue

        if in_doc:
            if delim in stripped:
                pos = line.find(delim)
                before = line[:pos].rstrip("\n")
                if needs_normalization(before):
                    before = normalize_text(before, "docstring")
                out.append(before + line[pos:])
                in_doc = False
                delim = ""
            else:
                body = line.rstrip("\n")
                if needs_normalization(body):
                    body = normalize_text(body, "docstring")
                out.append(body + ("\n" if line.endswith("\n") else ""))
            continue

        if "#" in line:
            i = line.find("#")
            prefix = line[:i]
            comment = line[i + 1 :].rstrip("\n")
            if needs_normalization(comment):
                comment = normalize_text(comment, "py")
            out.append(f"{prefix}# {comment.strip()}" + ("\n" if line.endswith("\n") else ""))
            continue

        out.append(line)

    return "".join(out)


def sanitize_js_css_html(content: str, suffix: str) -> str:
    def html_repl(m: re.Match[str]) -> str:
        text = m.group(1).strip()
        if needs_normalization(text):
            text = normalize_text(text, "html")
        return f"<!-- {text} -->"

    def block_repl(m: re.Match[str]) -> str:
        text = m.group(1).strip()
        ctx = "css" if suffix == ".css" else "js"
        if needs_normalization(text):
            text = normalize_text(text, ctx)
        return f"/* {text} */"

    def line_repl(m: re.Match[str]) -> str:
        prefix, text = m.group(1), m.group(2).strip()
        if needs_normalization(text):
            text = normalize_text(text, "js")
        return f"{prefix}// {text}"

    updated = re.sub(r"<!--\s*(.*?)\s*-->", html_repl, content, flags=re.DOTALL)
    updated = re.sub(r"/\*\s*(.*?)\s*\*/", block_repl, updated, flags=re.DOTALL)
    updated = re.sub(r"(^|\s)//\s*(.*)$", line_repl, updated, flags=re.MULTILINE)
    return updated


def tracked_sources() -> list[Path]:
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


def main() -> None:
    changed = 0
    for path in tracked_sources():
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        suffix = path.suffix.lower()
        if suffix == ".py":
            updated = sanitize_python(original)
        else:
            updated = sanitize_js_css_html(original, suffix)

        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1

    print(f"Normalized files: {changed}")


if __name__ == "__main__":
    main()
