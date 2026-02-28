"""
═══════════════════════════════════════════════════════════════
  lecture_summarizer.py — Конспект из лекций/PDF TARS v3
═══════════════════════════════════════════════════════════════

"Сделай конспект из этого файла"
"Перескажи лекцию в 10 пунктов"
"Выдели определения и формулы"

Поддерживает:
  - Текстовые файлы (.txt, .md)
  - PDF (через PyPDF2 или pdfplumber)
  - Word (.docx через python-docx)
  - Аудиозаписи лекций (.wav, .mp3 → Whisper)
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("Tars.LectureSummarizer")

_ROOT = Path(__file__).parent.parent
_NOTES_DIR = _ROOT / "data" / "notes"
_NOTES_DIR.mkdir(parents=True, exist_ok=True)


class LectureSummarizer:
    """
    Создаёт умные конспекты из файлов и аудио.
    
    Методы:
      - Extractive summarization (без GPT, на основе TF-IDF + позиции)
      - Выделение определений, формул, дат
      - Генерация flashcards для LearningHelper
    """
    
    def __init__(self, whisper_model=None):
        self.whisper = whisper_model
    
    def summarize_file(self, file_path: str, max_points: int = 10) -> str:
        """
        Конспект из файла.
        
        Определяет тип файла и вызывает нужный парсер.
        """
        path = Path(file_path)
        if not path.exists():
            return f"❌ Файл не найден: {file_path}"
        
        ext = path.suffix.lower()
        
        if ext in ('.txt', '.md', '.py', '.log'):
            text = self._read_text(file_path)
        elif ext == '.pdf':
            text = self._read_pdf(file_path)
        elif ext in ('.docx', '.doc'):
            text = self._read_docx(file_path)
        elif ext in ('.wav', '.mp3', '.ogg', '.m4a'):
            text = self._transcribe_audio(file_path)
        else:
            return f"❌ Формат {ext} не поддерживается. Поддержка: txt, pdf, docx, wav, mp3"
        
        if not text or len(text) < 50:
            return "❌ Файл пустой или слишком короткий для конспекта."
        
        return self._create_summary(text, path.name, max_points)
    
    def summarize_text(self, text: str, title: str = "Текст", max_points: int = 10) -> str:
        """Конспект из текста."""
        return self._create_summary(text, title, max_points)
    
    def extract_definitions(self, text: str) -> List[str]:
        """Выделить определения из текста."""
        definitions = []
        
        patterns = [
            r'(.+?)\s*[—–-]\s*это\s+(.+?)(?:\.|$)',        # X — это Y
            r'(.+?)\s*называется\s+(.+?)(?:\.|$)',            # X называется Y
            r'[Оо]пределение[:.]\s*(.+?)(?:\.|$)',            # Определение: ...
            r'(.+?)\s*is\s+(?:a|an|the)\s+(.+?)(?:\.|$)',    # X is a Y
            r'[Dd]efinition[:.]\s*(.+?)(?:\.|$)',             # Definition: ...
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    definition = " — ".join(str(m).strip() for m in match)
                else:
                    definition = str(match).strip()
                if len(definition) > 10 and definition not in definitions:
                    definitions.append(definition[:200])
        
        return definitions[:20]
    
    def extract_formulas(self, text: str) -> List[str]:
        """Выделить формулы из текста."""
        formulas = []
        
        patterns = [
            r'\$(.+?)\$',                      # LaTeX inline
            r'\\\[(.+?)\\\]',                  # LaTeX display
            r'[A-Za-z]+\s*=\s*[^,\n]{3,50}',  # X = expression
            r'∑|∫|∂|√|±|≤|≥|≠|∈|∉|⊂|∪|∩',    # Unicode math
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                formula = str(m).strip()
                if len(formula) > 2 and formula not in formulas:
                    formulas.append(formula[:100])
        
        return formulas[:15]
    
    def extract_dates_events(self, text: str) -> List[str]:
        """Выделить даты и события."""
        events = []
        
        # Разные форматы дат
        date_patterns = [
            r'(\d{1,2}[./]\d{1,2}[./]\d{2,4})\s*[—–-]?\s*(.{0,100})',
            r'(\d{4})\s*(?:год|г\.?)\s*[—–-]?\s*(.{0,100})',
            r'в\s+(\d{4})\s+(.{0,80})',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for date_str, event in matches:
                entry = f"{date_str.strip()} — {event.strip()}"
                if len(entry) > 10 and entry not in events:
                    events.append(entry[:150])
        
        return events[:10]
    
    def _create_summary(self, text: str, title: str, max_points: int) -> str:
        """Создать конспект из текста."""
        # Разбиваем на предложения
        sentences = self._split_sentences(text)
        
        if not sentences:
            return "❌ Не удалось разбить текст на предложения."
        
        # Оценка важности каждого предложения
        scores = self._score_sentences(sentences, text)
        
        # Топ-N предложений по важности, сохраняя порядок
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[:max_points])
        
        # Определения, формулы, даты
        definitions = self.extract_definitions(text)
        formulas = self.extract_formulas(text)
        dates = self.extract_dates_events(text)
        
        # Формируем конспект
        lines = [
            f"📋 Конспект: {title}",
            f"📊 Исходник: {len(text)} символов → {max_points} тезисов\n",
        ]
        
        lines.append("🎯 Ключевые тезисы:")
        for i, idx in enumerate(top_indices, 1):
            sent = sentences[idx].strip()
            if len(sent) > 200:
                sent = sent[:197] + "..."
            lines.append(f"  {i}. {sent}")
        
        if definitions:
            lines.append(f"\n📖 Определения ({len(definitions)}):")
            for d in definitions[:5]:
                lines.append(f"  • {d}")
        
        if formulas:
            lines.append(f"\n🔢 Формулы ({len(formulas)}):")
            for f in formulas[:5]:
                lines.append(f"  • {f}")
        
        if dates:
            lines.append(f"\n📅 Даты и события ({len(dates)}):")
            for e in dates[:5]:
                lines.append(f"  • {e}")
        
        # Тематический анализ (топ ключевые слова)
        keywords = self._extract_keywords(text, n=8)
        if keywords:
            lines.append(f"\n🏷 Темы: {', '.join(keywords)}")
        
        # Сохраняем конспект
        note_path = _NOTES_DIR / f"note_{datetime.now().strftime('%Y%m%d_%H%M')}_{title[:20]}.md"
        try:
            with open(note_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            lines.append(f"\n💾 Сохранено: {note_path.name}")
        except Exception:
            pass
        
        return "\n".join(lines)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Разбить текст на предложения."""
        # Убираем лишние пробелы и переносы
        text = re.sub(r'\s+', ' ', text)
        
        # Разбиваем по точкам, ? и !
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Фильтруем слишком короткие
        return [s for s in sentences if len(s.strip()) > 20]
    
    def _score_sentences(self, sentences: List[str], full_text: str) -> List[float]:
        """
        Оценка важности предложений.
        
        Факторы:
          1. TF-IDF ключевых слов
          2. Позиция (первое и последнее — важнее)
          3. Длина (средние предложения лучше)
          4. Наличие сигнальных слов
        """
        # TF-IDF
        word_freq = self._word_frequencies(full_text)
        n = len(sentences)
        
        scores = []
        for i, sent in enumerate(sentences):
            score = 0.0
            
            # 1. Частота слов
            words = sent.lower().split()
            if words:
                word_score = sum(word_freq.get(w, 0) for w in words) / len(words)
                score += word_score * 2.0
            
            # 2. Позиция (первые и последние предложения важнее)
            position = i / max(1, n - 1)
            if position < 0.1 or position > 0.9:
                score += 1.5
            elif position < 0.3:
                score += 0.8
            
            # 3. Длина (30-100 слов = оптимально)
            wlen = len(words) if words else 0
            if 5 <= wlen <= 30:
                score += 0.5
            
            # 4. Сигнальные слова
            signal_words = [
                'важно', 'главное', 'итог', 'вывод', 'результат',
                'определение', 'теорема', 'формула', 'правило', 'закон',
                'important', 'conclusion', 'result', 'therefore', 'thus',
                'key', 'main', 'primary', 'essential',
            ]
            for sw in signal_words:
                if sw in sent.lower():
                    score += 1.0
                    break
            
            scores.append(score)
        
        return scores
    
    def _word_frequencies(self, text: str) -> Dict[str, float]:
        """Частоты слов (нормализованные)."""
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'а', 'но', 'что', 'это', 'как',
            'я', 'мы', 'он', 'она', 'они', 'не', 'да', 'нет', 'для',
            'из', 'от', 'до', 'за', 'при', 'или', 'то', 'у', 'к', 'же',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'to', 'of', 'and', 'in', 'that', 'it', 'for', 'with', 'on',
        }
        
        words = re.findall(r'[а-яёa-z]{3,}', text.lower())
        freq = defaultdict(int)
        for w in words:
            if w not in stop_words:
                freq[w] += 1
        
        # Нормализация
        max_freq = max(freq.values()) if freq else 1
        return {w: c / max_freq for w, c in freq.items()}
    
    def _extract_keywords(self, text: str, n: int = 8) -> List[str]:
        """Извлечь ключевые слова."""
        freq = self._word_frequencies(text)
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:n]]
    
    def _read_text(self, path: str) -> str:
        """Чтение текстового файла."""
        encodings = ['utf-8', 'cp1251', 'latin-1']
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        return ""
    
    def _read_pdf(self, path: str) -> str:
        """Чтение PDF."""
        try:
            import PyPDF2
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                return "❌ Для PDF установи: pip install PyPDF2 или pip install pdfplumber"
    
    def _read_docx(self, path: str) -> str:
        """Чтение Word документа."""
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return "❌ Для DOCX установи: pip install python-docx"
    
    def _transcribe_audio(self, path: str) -> str:
        """Транскрибация аудиофайла через Whisper."""
        if self.whisper:
            try:
                result = self.whisper.transcribe(path)
                return result.get("text", "")
            except Exception as e:
                return f"❌ Ошибка транскрибации: {e}"
        
        # Fallback — пробуем whisper CLI
        try:
            import subprocess
            result = subprocess.run(
                ["whisper", path, "--language", "ru", "--output_format", "txt"],
                capture_output=True, text=True, timeout=300
            )
            txt_path = Path(path).with_suffix('.txt')
            if txt_path.exists():
                return txt_path.read_text(encoding='utf-8')
        except Exception:
            pass
        
        return "❌ Whisper не доступен. Установи: pip install openai-whisper"
    
    def generate_flashcards(self, text: str, topic: str = "general") -> List[Dict]:
        """
        Генерирует flashcards из текста для LearningHelper.
        
        Returns: [{"question": ..., "answer": ..., "topic": ...}, ...]
        """
        cards = []
        
        # Из определений
        definitions = self.extract_definitions(text)
        for defn in definitions:
            parts = defn.split(" — ", 1)
            if len(parts) == 2:
                cards.append({
                    "question": f"Что такое {parts[0].strip()}?",
                    "answer": parts[1].strip(),
                    "topic": topic,
                })
        
        # Из ключевых предложений
        sentences = self._split_sentences(text)
        scores = self._score_sentences(sentences, text)
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        
        for idx in ranked[:5]:
            sent = sentences[idx].strip()
            if len(sent) > 30:
                # Упрощённый вопрос из предложения
                cards.append({
                    "question": f"Объясни: {sent[:80]}...",
                    "answer": sent,
                    "topic": topic,
                })
        
        return cards[:10]
