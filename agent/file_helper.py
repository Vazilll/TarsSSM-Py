"""
═══════════════════════════════════════════════════════════════
  file_helper.py — Файловый помощник TARS v3
═══════════════════════════════════════════════════════════════

"Разбери Downloads по папкам"
"Найди файл про отчёт"
"Покажи дубликаты"
"Что я скачивал вчера?"
"""

import os
import hashlib
import logging
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("Tars.FileHelper")


# Категории файлов по расширению
FILE_CATEGORIES = {
    "Изображения": {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico', '.tiff'},
    "Документы": {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.md', '.epub'},
    "Таблицы": {'.xls', '.xlsx', '.csv', '.ods'},
    "Презентации": {'.ppt', '.pptx', '.odp'},
    "Видео": {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'},
    "Аудио": {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'},
    "Архивы": {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'},
    "Код": {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.rs', '.go', '.ts'},
    "Исполняемые": {'.exe', '.msi', '.bat', '.cmd', '.ps1', '.sh'},
    "Данные": {'.json', '.xml', '.yaml', '.yml', '.sql', '.db', '.sqlite'},
}


class FileHelper:
    """
    Файловый помощник — сортировка, поиск, анализ.
    """
    
    def __init__(self):
        self.home = Path.home()
        self.downloads = self.home / "Downloads"
        self.desktop = self.home / "Desktop"
    
    def sort_folder(self, folder: str = None, dry_run: bool = True) -> str:
        """
        Разобрать папку по категориям.
        dry_run=True → только показать что будет перемещено.
        """
        folder_path = Path(folder) if folder else self.downloads
        
        if not folder_path.exists():
            return f"❌ Папка не найдена: {folder_path}"
        
        moves = defaultdict(list)
        
        for f in folder_path.iterdir():
            if f.is_file():
                ext = f.suffix.lower()
                category = self._get_category(ext)
                if category:
                    moves[category].append(f)
        
        if not moves:
            return f"📁 Папка {folder_path.name} уже чистая!"
        
        lines = [f"📁 Сортировка {folder_path.name}:\n"]
        total = 0
        
        for category, files in sorted(moves.items()):
            size = sum(f.stat().st_size for f in files) / (1024 * 1024)
            lines.append(f"  📂 {category}/ ({len(files)} файлов, {size:.1f} MB)")
            for f in files[:3]:
                lines.append(f"    → {f.name}")
            if len(files) > 3:
                lines.append(f"    ... и ещё {len(files) - 3}")
            total += len(files)
        
        if dry_run:
            lines.append(f"\n📊 Итого: {total} файлов будет разобрано")
            lines.append("Скажи «разбери» чтобы выполнить")
        else:
            # Реально перемещаем
            moved = 0
            for category, files in moves.items():
                dest = folder_path / category
                dest.mkdir(exist_ok=True)
                for f in files:
                    try:
                        new_path = dest / f.name
                        if new_path.exists():
                            stem = f.stem
                            new_path = dest / f"{stem}_{int(datetime.now().timestamp())}{f.suffix}"
                        shutil.move(str(f), str(new_path))
                        moved += 1
                    except Exception as e:
                        logger.debug(f"Move error: {e}")
            lines.append(f"\n✅ Перемещено: {moved} файлов")
        
        return "\n".join(lines)
    
    def search(self, query: str, folder: str = None, 
               search_content: bool = False) -> str:
        """Поиск файлов по имени (и содержимому)."""
        search_dirs = [Path(folder)] if folder else [self.home]
        results = []
        query_lower = query.lower()
        
        for search_dir in search_dirs:
            try:
                for root, dirs, files in os.walk(str(search_dir)):
                    # Пропускаем системные папки
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in ('node_modules', '__pycache__', '.git', 'venv')]
                    
                    for fname in files:
                        if query_lower in fname.lower():
                            fpath = Path(root) / fname
                            try:
                                size = fpath.stat().st_size / 1024
                                mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
                                results.append((fpath, size, mtime))
                            except (OSError, PermissionError):
                                pass
                    
                    if len(results) >= 20:
                        break
            except PermissionError:
                continue
        
        if not results:
            return f"🔍 Не найдено: «{query}»"
        
        lines = [f"🔍 Найдено ({len(results)}):\n"]
        for fpath, size_kb, mtime in sorted(results, key=lambda x: -x[2].timestamp())[:15]:
            size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            date_str = mtime.strftime("%d.%m %H:%M")
            lines.append(f"  📄 {fpath.name} ({size_str}) — {date_str}")
            lines.append(f"     {fpath.parent}")
        
        return "\n".join(lines)
    
    def find_duplicates(self, folder: str = None) -> str:
        """Найти дубликаты файлов."""
        folder_path = Path(folder) if folder else self.downloads
        
        if not folder_path.exists():
            return f"❌ Папка не найдена: {folder_path}"
        
        # Группировка по размеру
        by_size = defaultdict(list)
        for f in folder_path.rglob("*"):
            if f.is_file():
                try:
                    size = f.stat().st_size
                    if size > 100:  # Игнорируем пустые/крошечные
                        by_size[size].append(f)
                except (OSError, PermissionError):
                    pass
        
        # Проверяем хеши для файлов одинакового размера
        duplicates = []
        for size, files in by_size.items():
            if len(files) < 2:
                continue
            
            hashes = defaultdict(list)
            for f in files:
                try:
                    h = hashlib.md5(f.read_bytes()[:8192]).hexdigest()
                    hashes[h].append(f)
                except (OSError, PermissionError):
                    pass
            
            for h, dups in hashes.items():
                if len(dups) >= 2:
                    duplicates.append(dups)
        
        if not duplicates:
            return f"✅ Дубликатов не найдено в {folder_path.name}"
        
        total_waste = 0
        lines = [f"🔍 Дубликаты в {folder_path.name}:\n"]
        for i, group in enumerate(duplicates[:10], 1):
            size = group[0].stat().st_size
            total_waste += size * (len(group) - 1)
            lines.append(f"  {i}. {group[0].name} ({size // 1024} KB) × {len(group)}")
            for f in group:
                lines.append(f"     {f}")
        
        lines.append(f"\n📊 Можно освободить: {total_waste // (1024*1024)} MB")
        return "\n".join(lines)
    
    def recent_files(self, folder: str = None, hours: int = 24) -> str:
        """Недавние файлы."""
        folder_path = Path(folder) if folder else self.downloads
        
        if not folder_path.exists():
            return f"❌ Папка не найдена: {folder_path}"
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for f in folder_path.iterdir():
            if f.is_file():
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime >= cutoff:
                        size = f.stat().st_size / 1024
                        recent.append((f, size, mtime))
                except (OSError, PermissionError):
                    pass
        
        if not recent:
            period = "сегодня" if hours <= 24 else f"за {hours} часов"
            return f"📁 Нет новых файлов {period} в {folder_path.name}"
        
        recent.sort(key=lambda x: -x[2].timestamp())
        
        period_str = "сегодня" if hours <= 24 else f"за {hours}ч"
        lines = [f"📁 Новые файлы {period_str} в {folder_path.name}:\n"]
        for f, size_kb, mtime in recent[:15]:
            size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            time_str = mtime.strftime("%H:%M")
            lines.append(f"  {time_str} — {f.name} ({size_str})")
        
        return "\n".join(lines)
    
    def folder_stats(self, folder: str = None) -> str:
        """Статистика папки."""
        folder_path = Path(folder) if folder else self.downloads
        
        if not folder_path.exists():
            return f"❌ Папка не найдена: {folder_path}"
        
        by_ext = defaultdict(lambda: {"count": 0, "size": 0})
        total_files = 0
        total_size = 0
        
        for f in folder_path.rglob("*"):
            if f.is_file():
                try:
                    size = f.stat().st_size
                    ext = f.suffix.lower() or "(без расширения)"
                    by_ext[ext]["count"] += 1
                    by_ext[ext]["size"] += size
                    total_files += 1
                    total_size += size
                except (OSError, PermissionError):
                    pass
        
        lines = [
            f"📊 {folder_path.name}: {total_files} файлов, "
            f"{total_size / (1024*1024):.1f} MB\n"
        ]
        
        sorted_exts = sorted(by_ext.items(), key=lambda x: -x[1]["size"])
        for ext, data in sorted_exts[:10]:
            size_mb = data["size"] / (1024 * 1024)
            lines.append(f"  {ext}: {data['count']} файлов ({size_mb:.1f} MB)")
        
        return "\n".join(lines)
    
    def _get_category(self, ext: str) -> Optional[str]:
        """Определить категорию файла."""
        for category, extensions in FILE_CATEGORIES.items():
            if ext in extensions:
                return category
        return None
