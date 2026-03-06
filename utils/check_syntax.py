"""Quick syntax check for all .py files in project."""
import py_compile, os, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SKIP = {'__pycache__', 'venv', '.git', 'hub', '.pytest_cache'}
errors = []
count = 0

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            count += 1
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(str(path))

print(f"Checked {count} .py files")
if errors:
    print(f"SYNTAX ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
else:
    print("ALL OK - zero syntax errors")
