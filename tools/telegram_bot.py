"""
═══════════════════════════════════════════════════════════════
  ТАРС Telegram Bot — Обучение из Telegram
═══════════════════════════════════════════════════════════════

ТАРС слушает Telegram, собирает сообщения, учится из них.

Использует python-telegram-bot (pip install python-telegram-bot).
Токен бота от @BotFather.

Возможности:
  - Принимает текстовые сообщения → сохраняет в LEANN
  - Принимает документы (.txt, .pdf) → извлекает текст → LEANN
  - Принимает голосовые → TODO: Whisper → текст → LEANN
  - Команды: /status, /search, /learn
"""

import os
import logging
import asyncio
import json
import time
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger("Tars.Telegram")

# Путь для сохранения данных из Telegram
_DATA_DIR = Path(__file__).parent.parent / "data" / "telegram"


class TarsTelegram:
    """
    Telegram-бот ТАРС для сбора данных и общения.
    
    Использование:
        bot = TarsTelegram(token="YOUR_BOT_TOKEN", memory=leann_index)
        await bot.start()
    """
    
    def __init__(self, token: str = "", 
                 memory=None,
                 on_message: Optional[Callable] = None,
                 allowed_users: Optional[list] = None):
        """
        Args:
            token: Telegram bot token от @BotFather
            memory: LeannIndex для сохранения данных
            on_message: Callback при новом сообщении
            allowed_users: Список разрешённых user_id (None = все)
        """
        self.token = token or os.environ.get("TARS_TELEGRAM_TOKEN", "")
        self.memory = memory
        self.on_message = on_message
        self.allowed_users = set(allowed_users) if allowed_users else None
        
        self.messages_collected = 0
        self.docs_collected = 0
        self._running = False
        self._data_dir = _DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # Файл для собранных данных (для последующего обучения)
        self._corpus_file = self._data_dir / "telegram_corpus.txt"
    
    def _check_user(self, user_id: int) -> bool:
        """Проверить доступ пользователя."""
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users
    
    async def start(self):
        """Запустить бота (polling mode)."""
        if not self.token:
            logger.error("Telegram: Нет токена! Установи TARS_TELEGRAM_TOKEN или передай token=")
            return
        
        try:
            from telegram import Update
            from telegram.ext import (
                Application, CommandHandler, MessageHandler, 
                filters, ContextTypes
            )
        except ImportError:
            logger.error("pip install python-telegram-bot")
            return
        
        app = Application.builder().token(self.token).build()
        
        # Команды
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("search", self._cmd_search))
        app.add_handler(CommandHandler("learn", self._cmd_learn))
        app.add_handler(CommandHandler("forget", self._cmd_forget))
        
        # Сообщения
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self._on_text
        ))
        app.add_handler(MessageHandler(
            filters.Document.ALL, self._on_document
        ))
        
        self._running = True
        logger.info("Telegram бот запущен")
        
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        
        # Ждём пока не остановят
        while self._running:
            await asyncio.sleep(1)
        
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
    
    def stop(self):
        """Остановить бота."""
        self._running = False
    
    # ═══════ Команды ═══════
    
    async def _cmd_start(self, update, context):
        if not self._check_user(update.effective_user.id):
            await update.message.reply_text("⛔ Нет доступа")
            return
        
        await update.message.reply_text(
            "🤖 ТАРС — Автономный ИИ\n\n"
            "Отправляй мне:\n"
            "• Текст — я запомню\n"
            "• Файлы (.txt) — я изучу\n"
            "• /search <запрос> — поиск в памяти\n"
            "• /learn <url> — изучить веб-страницу\n"
            "• /status — статус\n"
        )
    
    async def _cmd_status(self, update, context):
        if not self._check_user(update.effective_user.id):
            return
        
        mem_count = len(self.memory.texts) if self.memory else 0
        await update.message.reply_text(
            f"📊 ТАРС Status\n"
            f"🧠 Память: {mem_count} документов\n"
            f"💬 Собрано сообщений: {self.messages_collected}\n"
            f"📄 Собрано документов: {self.docs_collected}\n"
        )
    
    async def _cmd_search(self, update, context):
        if not self._check_user(update.effective_user.id):
            return
        
        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text("Использование: /search <запрос>")
            return
        
        if self.memory:
            results = await self.memory.search(query, top_k=3)
            if results:
                text = "🔍 Результаты:\n\n"
                for i, r in enumerate(results, 1):
                    snippet = r[:200] + "..." if len(r) > 200 else r
                    text += f"{i}. {snippet}\n\n"
                await update.message.reply_text(text[:4000])
            else:
                await update.message.reply_text("🤷 Ничего не найдено")
        else:
            await update.message.reply_text("⚠ Память не подключена")
    
    async def _cmd_learn(self, update, context):
        """Изучить веб-страницу."""
        if not self._check_user(update.effective_user.id):
            return
        
        url = " ".join(context.args) if context.args else ""
        if not url:
            await update.message.reply_text("Использование: /learn <url>")
            return
        
        # ═══ URL validation ═══
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            await update.message.reply_text("❌ Только HTTP/HTTPS URL")
            return
        
        await update.message.reply_text(f"📚 Загружаю: {url}")
        
        try:
            from tools.web_search import fetch_page_text
            text = await fetch_page_text(url, max_chars=5000)
            
            if text and self.memory:
                self.memory.add_document(text)
                self._save_to_corpus(text, source=f"web:{url}")
                self.docs_collected += 1
                
                await update.message.reply_text(
                    f"✅ Изучено: {len(text)} символов\n"
                    f"🧠 Память: {len(self.memory.texts)} документов"
                )
            else:
                await update.message.reply_text("❌ Не удалось загрузить страницу")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def _cmd_forget(self, update, context):
        if not self._check_user(update.effective_user.id):
            return
        await update.message.reply_text("🧠 Память не очищена (защита от случайного удаления)")
    
    # ═══════ Обработка сообщений ═══════
    
    async def _on_text(self, update, context):
        """Обработка текстового сообщения."""
        if not self._check_user(update.effective_user.id):
            return
        
        text = update.message.text.strip()
        if not text or len(text) < 10:
            return
        
        # Сохранить в LEANN
        if self.memory:
            user = update.effective_user
            tagged = f"[{user.first_name}] {text}"
            self.memory.add_document(tagged)
            self.messages_collected += 1
        
        # Сохранить в корпус для обучения
        self._save_to_corpus(text, source=f"tg:{update.effective_user.id}")
        
        # Callback
        if self.on_message:
            try:
                self.on_message(text, update.effective_user.id)
            except Exception as e:
                logger.debug(f"on_message callback error: {e}")
        
        # Ответить подтверждением (каждое 10-е)
        if self.messages_collected % 10 == 0:
            await update.message.reply_text(
                f"🧠 +{self.messages_collected} сообщений в памяти"
            )
    
    async def _on_document(self, update, context):
        """Обработка документа."""
        if not self._check_user(update.effective_user.id):
            return
        
        doc = update.message.document
        if not doc or doc.file_size > 10_000_000:  # 10 MB max
            return
        
        # Только текстовые файлы
        allowed_ext = ['.txt', '.md', '.csv', '.json', '.py', '.js']
        name = doc.file_name or ""
        if not any(name.endswith(ext) for ext in allowed_ext):
            await update.message.reply_text("⚠ Поддерживаю: .txt, .md, .csv, .json, .py")
            return
        
        try:
            file = await context.bot.get_file(doc.file_id)
            # ═══ Sanitize filename to prevent path traversal ═══
            safe_name = Path(name).name  # strip directory components
            file_path = self._data_dir / safe_name
            await file.download_to_drive(str(file_path))
            
            text = file_path.read_text(encoding='utf-8', errors='replace')
            
            if self.memory and text:
                # Разбить на чанки и добавить
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                for chunk in chunks[:100]:  # Макс 100 чанков из одного файла
                    if chunk.strip():
                        self.memory.add_document(chunk)
                
                self.docs_collected += 1
                self._save_to_corpus(text, source=f"tg_file:{name}")
                
                await update.message.reply_text(
                    f"📄 Изучено: {name}\n"
                    f"   {len(text)} символов, {len(chunks)} чанков\n"
                    f"🧠 Память: {len(self.memory.texts)} документов"
                )
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    # ═══════ Корпус для обучения ═══════
    
    def _save_to_corpus(self, text: str, source: str = ""):
        """Сохранить текст в корпус для дообучения модели."""
        with open(self._corpus_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[SRC:{source}] {text}\n")
    
    def get_corpus_size(self) -> int:
        """Размер собранного корпуса в байтах."""
        if self._corpus_file.exists():
            return self._corpus_file.stat().st_size
        return 0
    
    def get_corpus_path(self) -> str:
        """Путь к корпусу."""
        return str(self._corpus_file)
