import json

notebook_path = r'c:\Users\Public\Tarsfull\TarsSSM-Py\TARS_Colab.ipynb'

# Read with surrogatepass to handle emoji surrogate pairs in .ipynb
with open(notebook_path, 'r', encoding='utf-8', errors='surrogatepass') as f:
    raw = f.read()

nb = json.loads(raw)

test_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "#@title \\ud83e\\uddea **\\u0422\\u0415\\u0421\\u0422** \\u2014 \\u043f\\u0440\\u043e\\u0432\\u0435\\u0440\\u043a\\u0430 \\u0447\\u0442\\u043e \\u0432\\u0441\\u0451 \\u0440\\u0430\\u0431\\u043e\\u0442\\u0430\\u0435\\u0442 (~5 \\u043c\\u0438\\u043d) { display-mode: \"form\" }\n",
        "# ============================================================\n",
        "# Quick test: MinGRU + Mamba-2\n",
        "# ============================================================\n",
        "\n",
        "import os, sys, time, subprocess, shutil\n",
        "from pathlib import Path\n",
        "\n",
        "# ==== 1. GOOGLE DRIVE ====\n",
        "print('=' * 60)\n",
        "print('  TARS — Test Run')\n",
        "print('=' * 60)\n",
        "print()\n",
        "\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "DRIVE_MODELS = Path('/content/drive/MyDrive/TarsModels')\n",
        "DRIVE_MODELS.mkdir(parents=True, exist_ok=True)\n",
        "print()\n",
        "\n",
        "# ==== 2. GPU CHECK ====\n",
        "!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n",
        "print()\n",
        "\n",
        "# ==== 3. CLONE / UPDATE ====\n",
        "os.chdir('/content')\n",
        "if os.path.exists('TarsSSM-Py'):\n",
        "    os.chdir('/content/TarsSSM-Py')\n",
        "    !git pull --rebase 2>/dev/null || true\n",
        "else:\n",
        "    !git clone https://github.com/Vazilll/TarsSSM-Py.git\n",
        "    os.chdir('/content/TarsSSM-Py')\n",
        "\n",
        "ROOT = Path('/content/TarsSSM-Py')\n",
        "print(f'  Root: {ROOT}')\n",
        "print()\n",
        "\n",
        "# ==== 4. INSTALL DEPS ====\n",
        "print('  Installing deps...')\n",
        "!pip install -q torch einops numpy tqdm sentencepiece 2>/dev/null\n",
        "print()\n",
        "\n",
        "# ==== 5. RUN QUICK TEST ====\n",
        "print('=' * 60)\n",
        "print('  Running tests: MinGRU + Mamba-2')\n",
        "print('=' * 60)\n",
        "print(flush=True)\n",
        "\n",
        "_t0 = time.time()\n",
        "\n",
        "env = os.environ.copy()\n",
        "env['PYTHONUNBUFFERED'] = '1'\n",
        "\n",
        "process = subprocess.Popen(\n",
        "    [sys.executable, '-u', str(ROOT / 'training' / 'quick_test.py')],\n",
        "    cwd=str(ROOT),\n",
        "    stdout=subprocess.PIPE,\n",
        "    stderr=subprocess.STDOUT,\n",
        "    env=env,\n",
        "    bufsize=1,\n",
        "    universal_newlines=True,\n",
        ")\n",
        "\n",
        "for line in process.stdout:\n",
        "    print(line, end='', flush=True)\n",
        "\n",
        "returncode = process.wait()\n",
        "_elapsed = time.time() - _t0\n",
        "\n",
        "print()\n",
        "print('=' * 60)\n",
        "if returncode == 0:\n",
        "    print(f'  ALL TESTS PASSED in {_elapsed:.0f}s')\n",
        "    print(f'  -> System ready for training!')\n",
        "    print(f'  -> Run next cell for full training')\n",
        "else:\n",
        "    print(f'  TESTS FAILED (code {returncode})')\n",
        "    print(f'     Time: {_elapsed:.0f}s')\n",
        "    print(f'     Check errors above')\n",
        "print('=' * 60)"
    ]
}

# Insert test cell after the markdown header (index 0), before the main training cell (index 1)
has_test = False
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'Test Run' in src or '\\u0422\\u0415\\u0421\\u0422' in src:
        has_test = True
        break

if has_test:
    print("Test cell already exists, replacing...")
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        if ('Test Run' in src or '\\u0422\\u0415\\u0421\\u0422' in src) and c['cell_type'] == 'code':
            nb['cells'][i] = test_cell
            break
else:
    print("Adding test cell after markdown header...")
    nb['cells'].insert(1, test_cell)

# Write back using ensure_ascii=True to avoid surrogate issues
out = json.dumps(nb, ensure_ascii=True, indent=4)
with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(out)

print(f"Done! Notebook saved: {notebook_path}")
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))[:60]
    print(f"  Cell {i}: [{c['cell_type']}] {src}")
