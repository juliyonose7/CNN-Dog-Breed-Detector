import os, re

patterns = {
    'API Key': re.compile(r'(?i)(api[_-]?key|token|secret)\s*[=:]\s*["\'][A-Za-z0-9+/=_\-.]{8,}'),
    'Hardcoded URL (non-localhost)': re.compile(r'https?://(?!localhost|127\.0\.0\.1)[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),
    'Email': re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),
    'Local path': re.compile(r'[A-Za-z]:\\\\|C:/Users/|/home/[a-z]+/'),
}
skip_dirs = {'.venv', '__pycache__', '.git', 'node_modules'}
exts = {'.js', '.ts', '.jsx', '.tsx', '.html', '.css'}
skip_files = {'package-lock.json'}

found = False
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for fname in files:
        if fname in skip_files:
            continue
        ext = os.path.splitext(fname)[1]
        if ext in exts:
            path = os.path.join(root, fname)
            with open(path, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    s = line.strip()
                    if s.startswith('//') or s.startswith('#'):
                        continue
                    for label, pat in patterns.items():
                        if pat.search(line):
                            print(f'[{label}] {path}:{i}: {s[:110]}')
                            found = True
if not found:
    print('No sensitive data found in frontend files.')
