"""
═══════════════════════════════════════════════════════════════
  setup_gdrive.py — Подключение Google Drive на headless сервере
═══════════════════════════════════════════════════════════════

Устанавливает rclone, авторизует Google Drive, синхронизирует данные.
Работает на сервере БЕЗ GUI — авторизация через браузер на другом ПК.

Использование:
  python setup_gdrive.py setup          # Первоначальная настройка
  python setup_gdrive.py status         # Проверить подключение
  python setup_gdrive.py sync           # Синхронизировать данные
  python setup_gdrive.py sync-models    # Выгрузить модели на Drive
  python setup_gdrive.py mount          # Смонтировать Drive как папку

═══════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
import shutil
import json
import time
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Google Drive paths (настраиваемые)
GDRIVE_REMOTE = "gdrive"                    # Имя rclone remote
GDRIVE_DATA = "tars_training/data"           # Папка с данными на Drive
GDRIVE_MODELS = "tars_training/models"       # Папка с моделями на Drive
GDRIVE_MOUNT = ROOT / "gdrive_mount"         # Куда монтировать

IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"


def print_banner():
    print()
    print("═" * 60)
    print("  ☁️  TARS Google Drive Setup")
    print("═" * 60)
    print()


def run_cmd(cmd, check=True, capture=False):
    """Run shell command."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    try:
        r = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check,
        )
        return r.stdout.strip() if capture else r.returncode == 0
    except Exception as e:
        if not check:
            return "" if capture else False
        raise


# ═══════════════════════════════════════════
# 1. Install rclone
# ═══════════════════════════════════════════

def install_rclone():
    """Установить rclone если не установлен."""
    
    # Check if already installed
    if shutil.which("rclone"):
        version = run_cmd(["rclone", "version", "--check"], capture=True, check=False)
        if not version:
            version = run_cmd(["rclone", "version"], capture=True, check=False)
        ver_line = version.split('\n')[0] if version else "installed"
        print(f"  ✅ rclone: {ver_line}")
        return True
    
    print("  📦 Установка rclone...")
    
    if IS_LINUX:
        # Official install script (works on any Linux)
        try:
            subprocess.run(
                ["bash", "-c", "curl -fsSL https://rclone.org/install.sh | sudo bash"],
                check=True,
            )
            print("  ✅ rclone установлен")
            return True
        except Exception:
            # Fallback: pip
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "rclone"], check=True)
                print("  ✅ rclone установлен (via pip)")
                return True
            except Exception as e:
                print(f"  ❌ Не удалось установить rclone: {e}")
                print("  Установите вручную: https://rclone.org/install/")
                return False
    
    elif IS_WINDOWS:
        print("  🪟 На Windows скачайте rclone:")
        print("     https://rclone.org/downloads/")
        print("     Или: winget install Rclone.Rclone")
        try:
            subprocess.run(["winget", "install", "Rclone.Rclone", "--accept-package-agreements"], check=True)
            print("  ✅ rclone установлен через winget")
            return True
        except Exception:
            print("  ⚠ Установите rclone вручную и перезапустите")
            return False
    
    return False


# ═══════════════════════════════════════════
# 2. Configure Google Drive (headless auth)
# ═══════════════════════════════════════════

def setup_gdrive():
    """Настройка Google Drive — headless авторизация."""
    
    print_banner()
    
    # Step 1: Install rclone
    if not install_rclone():
        return False
    
    # Step 2: Check if already configured
    remotes = run_cmd(["rclone", "listremotes"], capture=True, check=False)
    if f"{GDRIVE_REMOTE}:" in (remotes or ""):
        print(f"\n  ✅ Remote '{GDRIVE_REMOTE}' уже настроен")
        print(f"  Для переподключения: rclone config delete {GDRIVE_REMOTE}")
        return verify_connection()
    
    # Step 3: Headless auth flow
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  АВТОРИЗАЦИЯ GOOGLE DRIVE (headless)                │")
    print("  │                                                     │")
    print("  │  Поскольку на сервере нет браузера, авторизация     │")
    print("  │  происходит через другой ПК:                        │")
    print("  │                                                     │")
    print("  │  1. Будет показана ссылка — откройте её в браузере  │")
    print("  │     на ДРУГОМ компьютере (телефон тоже подойдёт)    │")
    print("  │  2. Войдите в Google аккаунт                        │")
    print("  │  3. Скопируйте verification code                   │")
    print("  │  4. Вставьте его в терминал                         │")
    print("  │                                                     │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    
    # Run rclone config interactively
    print("  Запускаю rclone config...\n")
    print("  ─" * 30)
    print("  Ответьте на вопросы rclone:")
    print(f"    n/s/q> n                    (new remote)")
    print(f"    name>  {GDRIVE_REMOTE}      (имя)")
    print(f"    Storage> drive              (Google Drive)")
    print(f"    client_id> (пустое, Enter)")
    print(f"    client_secret> (пустое, Enter)")
    print(f"    scope> 1                    (Full access)")
    print(f"    root_folder_id> (пустое)")
    print(f"    service_account_file> (пустое)")
    print(f"    Edit advanced config> n")
    print(f"    Use auto config> n          (ВАЖНО! headless = n)")
    print(f"    → Появится ссылка — откройте в браузере")
    print(f"    → Вставьте verification code")
    print(f"    team_drive> n")
    print(f"    y/n> y                      (подтвердить)")
    print(f"    q> q                        (выход)")
    print("  ─" * 30)
    print()
    
    subprocess.run(["rclone", "config"])
    
    # Verify
    return verify_connection()


# ═══════════════════════════════════════════
# 3. Verify connection
# ═══════════════════════════════════════════

def verify_connection():
    """Проверить подключение к Google Drive."""
    
    print("\n  🔍 Проверка подключения...")
    
    # Check remote exists
    remotes = run_cmd(["rclone", "listremotes"], capture=True, check=False)
    if not remotes or f"{GDRIVE_REMOTE}:" not in remotes:
        print(f"  ❌ Remote '{GDRIVE_REMOTE}' не найден")
        print(f"     Запустите: python setup_gdrive.py setup")
        return False
    
    print(f"  ✅ Remote: {GDRIVE_REMOTE}")
    
    # Check access
    try:
        about = run_cmd(
            ["rclone", "about", f"{GDRIVE_REMOTE}:"],
            capture=True, check=True,
        )
        print(f"  ✅ Доступ к Drive подтверждён")
        for line in about.split('\n'):
            if line.strip():
                print(f"     {line.strip()}")
    except Exception as e:
        print(f"  ❌ Нет доступа: {e}")
        return False
    
    # Check/create folders
    for folder in [GDRIVE_DATA, GDRIVE_MODELS]:
        try:
            run_cmd(["rclone", "mkdir", f"{GDRIVE_REMOTE}:{folder}"], check=True)
            print(f"  ✅ Папка: {folder}")
        except Exception:
            print(f"  ⚠ Не удалось создать: {folder}")
    
    # List data files
    print(f"\n  📂 Файлы в {GDRIVE_DATA}:")
    try:
        ls = run_cmd(
            ["rclone", "lsf", f"{GDRIVE_REMOTE}:{GDRIVE_DATA}",
             "--format", "sp", "--separator", "  "],
            capture=True, check=True,
        )
        if ls:
            total_mb = 0
            for line in ls.split('\n'):
                parts = line.strip().split('  ')
                if len(parts) >= 2:
                    size_str, name = parts[0], parts[1]
                    try:
                        size_bytes = int(size_str)
                        mb = size_bytes / (1024*1024)
                        total_mb += mb
                        print(f"    {name:40s} {mb:8.1f} MB")
                    except ValueError:
                        print(f"    {line.strip()}")
                elif line.strip():
                    print(f"    {line.strip()}")
            if total_mb > 0:
                print(f"    {'─' * 50}")
                print(f"    Итого: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        else:
            print("    (пусто — загрузите данные на Drive)")
            print(f"    Загрузите файлы в Google Drive → {GDRIVE_DATA}/")
    except Exception as e:
        print(f"    ⚠ {e}")
    
    print()
    return True


# ═══════════════════════════════════════════
# 4. Sync data FROM Google Drive → local
# ═══════════════════════════════════════════

def sync_data():
    """Синхронизировать данные с Google Drive → локально."""
    
    print_banner()
    print("  📥 Синхронизация: Google Drive → локальные данные\n")
    
    DATA.mkdir(parents=True, exist_ok=True)
    
    print(f"  Источник:  {GDRIVE_REMOTE}:{GDRIVE_DATA}")
    print(f"  Цель:      {DATA}")
    print()
    
    # Sync with progress
    try:
        subprocess.run([
            "rclone", "sync",
            f"{GDRIVE_REMOTE}:{GDRIVE_DATA}",
            str(DATA),
            "--progress",
            "--transfers", "4",       # 4 параллельных загрузки
            "--checkers", "8",
            "--drive-chunk-size", "64M",  # Большие чанки для скорости
            "--verbose",
        ], check=True)
        
        # Show result
        total_mb = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file()) / (1024*1024)
        file_count = sum(1 for f in DATA.rglob("*") if f.is_file())
        print(f"\n  ✅ Синхронизировано: {file_count} файлов, {total_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Ошибка синхронизации: {e}")
        return False


# ═══════════════════════════════════════════
# 5. Sync models TO Google Drive (backup)
# ═══════════════════════════════════════════

def sync_models():
    """Выгрузить модели на Google Drive (бэкап)."""
    
    print_banner()
    print("  📤 Выгрузка моделей: локально → Google Drive\n")
    
    tars_v3 = MODELS / "tars_v3"
    if not tars_v3.exists():
        print("  ⚠ models/tars_v3/ не найдена — сначала обучите модель")
        return False
    
    print(f"  Источник:  {tars_v3}")
    print(f"  Цель:      {GDRIVE_REMOTE}:{GDRIVE_MODELS}")
    print()
    
    try:
        subprocess.run([
            "rclone", "sync",
            str(tars_v3),
            f"{GDRIVE_REMOTE}:{GDRIVE_MODELS}",
            "--progress",
            "--transfers", "2",
            "--drive-chunk-size", "128M",
        ], check=True)
        
        total_mb = sum(f.stat().st_size for f in tars_v3.rglob("*") if f.is_file()) / (1024*1024)
        print(f"\n  ✅ Модели выгружены: {total_mb:.0f} MB")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Ошибка: {e}")
        return False


# ═══════════════════════════════════════════
# 6. Mount Google Drive as folder (FUSE)
# ═══════════════════════════════════════════

def mount_drive():
    """Смонтировать Google Drive как локальную папку."""
    
    print_banner()
    
    if not IS_LINUX:
        print("  ⚠ Монтирование поддерживается только на Linux (FUSE)")
        print("  На Windows используйте: python setup_gdrive.py sync")
        return False
    
    GDRIVE_MOUNT.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 Монтирование {GDRIVE_REMOTE}:{GDRIVE_DATA} → {GDRIVE_MOUNT}")
    print(f"  Для отмонтирования: fusermount -u {GDRIVE_MOUNT}")
    print()
    
    try:
        # Mount in background with caching
        subprocess.Popen([
            "rclone", "mount",
            f"{GDRIVE_REMOTE}:{GDRIVE_DATA}",
            str(GDRIVE_MOUNT),
            "--vfs-cache-mode", "full",     # Полное кеширование
            "--vfs-cache-max-size", "20G",  # Кеш до 20GB
            "--vfs-read-chunk-size", "64M",
            "--vfs-read-chunk-size-limit", "512M",
            "--buffer-size", "256M",
            "--daemon",                      # В фоне
        ])
        
        time.sleep(2)
        
        # Verify mount
        if GDRIVE_MOUNT.is_mount() or any(GDRIVE_MOUNT.iterdir()):
            files = list(GDRIVE_MOUNT.glob("*"))
            print(f"  ✅ Смонтировано! {len(files)} файлов видно")
            print(f"\n  Теперь запускайте обучение:")
            print(f"    python local_train.py --data-dir {GDRIVE_MOUNT}")
            return True
        else:
            print("  ⚠ Монтирование в процессе... Подождите 5 секунд и проверьте:")
            print(f"    ls {GDRIVE_MOUNT}")
            return True
            
    except Exception as e:
        print(f"  ❌ {e}")
        print("  Установите fuse: sudo apt install fuse3")
        return False


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def print_help():
    print_banner()
    print("  Команды:")
    print()
    print("    python setup_gdrive.py setup        Первоначальная настройка")
    print("    python setup_gdrive.py status        Проверить подключение")
    print("    python setup_gdrive.py sync          Скачать данные с Drive")
    print("    python setup_gdrive.py sync-models   Выгрузить модели на Drive")
    print("    python setup_gdrive.py mount         Смонтировать Drive (Linux)")
    print()
    print("  Полный сценарий:")
    print("    1. python setup_gdrive.py setup      ← авторизация (1 раз)")
    print("    2. Загрузите данные в Google Drive → tars_training/data/")
    print("    3. python setup_gdrive.py sync        ← скачать на сервер")
    print("    4. python local_train.py              ← обучение")
    print("    5. python setup_gdrive.py sync-models ← бэкап моделей")
    print()
    print("  Или через монтирование (Linux):")
    print("    1. python setup_gdrive.py setup")
    print("    2. python setup_gdrive.py mount")
    print("    3. python local_train.py --data-dir gdrive_mount/")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "setup":
        ok = setup_gdrive()
        sys.exit(0 if ok else 1)
    elif cmd == "status":
        ok = verify_connection()
        sys.exit(0 if ok else 1)
    elif cmd == "sync":
        ok = sync_data()
        sys.exit(0 if ok else 1)
    elif cmd in ("sync-models", "backup"):
        ok = sync_models()
        sys.exit(0 if ok else 1)
    elif cmd == "mount":
        ok = mount_drive()
        sys.exit(0 if ok else 1)
    else:
        print_help()
