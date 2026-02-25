"""
═══════════════════════════════════════════════════════════════
  Wikipedia Corpus Downloader — Массивный корпус из Википедии
═══════════════════════════════════════════════════════════════

Скачивает ТЫСЯЧИ статей с русской Википедии для обучения.
По умолчанию: 5000 статей (~50-100 MB текста).

Использование:
  python training/download_wiki.py                     # 5000 статей
  python training/download_wiki.py --count 10000       # 10000 статей
  python training/download_wiki.py --output data/wiki_ru.txt
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════════
# 300+ seed topics для гарантированно качественных статей
# ═══════════════════════════════════════════════════════════════

SEED_TOPICS = [
    # ─── Точные науки ───
    "Математика", "Физика", "Химия", "Астрономия", "Геология",
    "Квантовая механика", "Теория относительности", "Термодинамика",
    "Электродинамика", "Оптика", "Акустика", "Ядерная физика",
    "Физика элементарных частиц", "Теория струн", "Космология",
    "Органическая химия", "Неорганическая химия", "Биохимия",
    "Электрохимия", "Полимеры", "Катализ",
    
    # ─── Математика (углубленно) ───
    "Линейная алгебра", "Математический анализ", "Дифференциальное уравнение",
    "Теория вероятностей", "Математическая статистика", "Теория графов",
    "Комбинаторика", "Топология", "Группа (математика)", "Кольцо (математика)",
    "Матрица (математика)", "Интеграл", "Производная", "Ряд Фурье",
    "Теория чисел", "Алгебра", "Геометрия", "Тригонометрия",
    "Логарифм", "Дискретная математика", "Теория множеств",
    "Функциональный анализ", "Комплексное число", "Вектор (математика)",
    "Тензор", "Гильбертово пространство", "Вероятность",
    "Нормальное распределение", "Байесовская статистика",
    "Марковская цепь", "Случайный процесс", "Энтропия (теория информации)",
    
    # ─── Информатика и программирование ───
    "Информатика", "Алгоритм", "Структура данных", "Компьютер",
    "Программирование", "Python (язык программирования)", "Язык программирования",
    "Операционная система", "Криптография", "Интернет", "Компьютерная сеть",
    "База данных", "Процессор", "Оперативная память", "Жёсткий диск",
    "Файловая система", "Компилятор", "Интерпретатор (программирование)",
    "Объектно-ориентированное программирование", "Функциональное программирование",
    "Рекурсия", "Сортировка", "Хеш-таблица", "Двоичное дерево",
    "Граф (математика)", "Машина Тьюринга", "Теория вычислимости",
    "Сложность алгоритма", "NP-полная задача", "Параллельные вычисления",
    "Распределённые вычисления", "Облачные вычисления",
    "Linux", "Windows", "Unix", "Git", "Docker (программное обеспечение)",
    "JavaScript", "Java", "C (язык программирования)", "C++",
    "Rust (язык программирования)", "Go (язык программирования)",
    "SQL", "HTML", "TCP/IP", "HTTP", "API",
    
    # ─── Искусственный интеллект ───
    "Искусственный интеллект", "Машинное обучение", "Нейронная сеть",
    "Глубокое обучение", "Обучение с подкреплением", "Компьютерное зрение",
    "Обработка естественного языка", "Распознавание речи",
    "Генеративно-состязательная сеть", "Рекуррентная нейронная сеть",
    "Свёрточная нейронная сеть", "Трансформер (модель машинного обучения)",
    "Обратное распространение ошибки", "Градиентный спуск",
    "Переобучение (машинное обучение)", "Регуляризация (математика)",
    "Кластерный анализ", "Классификация", "Регрессия (математика)",
    "Нечёткая логика", "Экспертная система", "Генетический алгоритм",
    "Робототехника", "Автономный автомобиль",
    
    # ─── Биология ───
    "Биология", "Эволюция", "Генетика", "ДНК", "РНК", "Белок",
    "Клетка (биология)", "Ген", "Хромосома", "Мутация",
    "Естественный отбор", "Экология", "Экосистема",
    "Фотосинтез", "Метаболизм", "Митоз", "Мейоз",
    "Микробиология", "Вирус", "Бактерия", "Иммунитет",
    "Нейронаука", "Головной мозг", "Нервная система",
    "Сердечно-сосудистая система", "Пищеварительная система",
    "Анатомия человека", "Физиология", "Медицина",
    
    # ─── Космос ───
    "Солнечная система", "Солнце", "Луна", "Земля", "Марс",
    "Юпитер", "Сатурн", "Венера", "Меркурий", "Нептун", "Уран",
    "Вселенная", "Галактика", "Млечный Путь", "Чёрная дыра",
    "Нейтронная звезда", "Сверхновая звезда", "Большой взрыв",
    "Тёмная материя", "Тёмная энергия", "Экзопланета",
    "Космонавтика", "Юрий Гагарин", "Международная космическая станция",
    "SpaceX", "NASA", "Роскосмос", "Орбита", "Гравитация",
    
    # ─── Физика и инженерия ───
    "Электричество", "Магнетизм", "Электромагнитное излучение",
    "Радиоволны", "Микроволновое излучение", "Инфракрасное излучение",
    "Ультрафиолетовое излучение", "Рентгеновское излучение",
    "Лазер", "Полупроводник", "Транзистор", "Микросхема",
    "Электродвигатель", "Генератор электрического тока",
    "Ядерная энергетика", "Солнечная энергетика", "Ветроэнергетика",
    "Аккумулятор", "Конденсатор", "Резистор", "Диод",
    "Сверхпроводимость", "Нанотехнология", "Квантовый компьютер",
    
    # ─── История ───
    "Россия", "История России", "Москва", "Санкт-Петербург",
    "Киевская Русь", "Российская империя", "СССР",
    "Пётр I", "Екатерина II", "Иван Грозный",
    "Вторая мировая война", "Первая мировая война",
    "Холодная война", "Гражданская война в России",
    "Великая Отечественная война", "День Победы",
    "Древний Рим", "Древняя Греция", "Древний Египет",
    "Средние века", "Возрождение", "Промышленная революция",
    "Французская революция", "Октябрьская революция",
    
    # ─── География ───
    "Европа", "Азия", "Африка", "Северная Америка", "Южная Америка",
    "Австралия", "Антарктида", "Тихий океан", "Атлантический океан",
    "Байкал", "Волга", "Сибирь", "Урал", "Кавказ",
    "Китай", "Япония", "Индия", "Германия", "Франция",
    "Великобритания", "США", "Канада", "Бразилия",
    "Климат", "Пустыня", "Тропический лес", "Тайга", "Тундра",
    
    # ─── Культура и искусство ───
    "Литература", "Поэзия", "Роман (литература)", "Философия",
    "Пушкин, Александр Сергеевич", "Толстой, Лев Николаевич",
    "Достоевский, Фёдор Михайлович", "Чехов, Антон Павлович",
    "Музыка", "Классическая музыка", "Рок-музыка", "Джаз",
    "Кинематограф", "Архитектура", "Живопись", "Скульптура",
    "Логика", "Этика", "Эстетика", "Метафизика",
    "Аристотель", "Платон", "Кант, Иммануил", "Ницше, Фридрих",
    
    # ─── Естествознание ───
    "Вода", "Воздух", "Атмосфера Земли", "Энергия",
    "Звук", "Свет", "Температура", "Давление",
    "Золото", "Серебро", "Железо", "Углерод", "Кислород", "Водород",
    "Человек", "Мозг", "Сердце", "Кровь", "Зрение", "Слух",
    "Питание", "Витамины", "Сон", "Спорт",
    
    # ─── Технологии и изобретения ───
    "Паровой двигатель", "Двигатель внутреннего сгорания",
    "Электричество", "Телефон", "Телевидение", "Радио",
    "Атомная бомба", "Реактивный двигатель", "Самолёт",
    "Автомобиль", "Поезд", "Ракета", "Спутник",
    "Печатный станок", "Бумага", "Компас", "Порох",
    "3D-печать", "Блокчейн", "Криптовалюта", "Виртуальная реальность",
    
    # ─── Математики и учёные ───
    "Эйлер, Леонард", "Гаусс, Карл Фридрих", "Ньютон, Исаак",
    "Эйнштейн, Альберт", "Тесла, Никола", "Менделеев, Дмитрий Иванович",
    "Ломоносов, Михаил Васильевич", "Курчатов, Игорь Васильевич",
    "Колмогоров, Андрей Николаевич", "Тьюринг, Алан",
    "Дарвин, Чарлз", "Павлов, Иван Петрович", "Мария Кюри",
    "Галилей, Галилео", "Архимед", "Пифагор", "Евклид",
    "Фарадей, Майкл", "Максвелл, Джеймс Клерк",
    "Планк, Макс", "Бор, Нильс", "Гейзенберг, Вернер",
    "Шрёдингер, Эрвин", "Дирак, Поль", "Ферми, Энрико",
    "Хокинг, Стивен", "Фейнман, Ричард",
    
    # ─── Шахматы, игры, спорт ───
    "Шахматы", "Го (игра)", "Футбол", "Хоккей", "Теннис",
    "Олимпийские игры", "Чемпионат мира по футболу",
    
    # ─── Экономика и общество ───
    "Экономика", "Рынок", "Деньги", "Банк", "Инфляция",
    "Капитализм", "Социализм", "Демократия", "Право",
    "Образование", "Университет", "Наука",
]


def fetch_wiki_article(title: str) -> tuple:
    """Загружает текст статьи из русской Википедии. Returns (title, text)."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
        "exsectionformat": "plain",
    }
    url = "https://ru.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TARS-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                return (title, "")
            text = page.get("extract", "")
            return (title, clean_wiki_text(text))
    except Exception:
        return (title, "")


def fetch_random_titles(count: int = 500) -> list:
    """Получает случайные заголовки статей (до 500 за раз)."""
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": "0",
        "rnlimit": str(min(count, 500)),
        "format": "json",
    }
    url = "https://ru.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TARS-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        return [item["title"] for item in data.get("query", {}).get("random", [])]
    except Exception:
        return []


def fetch_links_from(title: str, limit: int = 50) -> list:
    """Получает ссылки из статьи (для обхода графа Википедии)."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "pllimit": str(limit),
        "plnamespace": "0",
        "format": "json",
    }
    url = "https://ru.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TARS-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        pages = data.get("query", {}).get("pages", {})
        links = []
        for page in pages.values():
            for link in page.get("links", []):
                links.append(link["title"])
        return links
    except Exception:
        return []


def clean_wiki_text(text: str) -> str:
    """Очищает текст статьи от мусора."""
    for section in ["== Примечания ==", "== Ссылки ==", "== Литература ==", 
                     "== См. также ==", "== Библиография ==", "== Источники =="]:
        idx = text.find(section)
        if idx > 0:
            text = text[:idx]
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\s+\n', '\n\n', text)
    text = re.sub(r'={2,}\s*(.+?)\s*={2,}', r'\1.', text)
    
    return text.strip()


def download_corpus(count: int = 100000, output_path: str = None) -> str:
    """
    Скачивает МАССИВНЫЙ корпус статей.
    
    Стратегия:
      1. Seed topics (300+ гарантированно качественных статей)
      2. Link crawl (ссылки из seed → ещё 2000+ статей)
      3. Random fill (случайные статьи до нужного count)
    
    Args:
        count: Количество статей (default: 5000)
        output_path: Путь для сохранения
    
    Returns:
        Текст корпуса
    """
    if output_path is None:
        output_path = str(ROOT / "data" / "wiki_ru.txt")
    
    # Проверяем кеш
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        if size > 1_000_000:  # > 1MB = уже скачано
            print(f"[Wiki] ✓ Корпус уже существует: {output_path} ({size / 1024 / 1024:.1f} MB)")
            with open(output_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"[Wiki] ═══ Скачиваю {count} статей из русской Википедии ═══")
    print(f"[Wiki] Это может занять 10-30 минут. Результат будет кеширован.")
    articles = []
    titles_done = set()
    
    # ═══════════════════════════════════════
    # Фаза 1: Seed topics (300+ статей)
    # ═══════════════════════════════════════
    print(f"\n[Wiki] Фаза 1/3: {len(SEED_TOPICS)} ключевых статей...")
    
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_wiki_article, t): t for t in SEED_TOPICS}
        done = 0
        for future in as_completed(futures):
            title, text = future.result()
            done += 1
            if text and len(text) > 200:
                articles.append(text)
                titles_done.add(title)
            if done % 50 == 0:
                print(f"  [{done}/{len(SEED_TOPICS)}] → {len(articles)} статей")
    
    print(f"  Фаза 1 завершена: {len(articles)} статей")
    
    # ═══════════════════════════════════════
    # Фаза 2: Link crawl (из seed статей)
    # ═══════════════════════════════════════
    if len(articles) < count:
        remaining = count - len(articles)
        print(f"\n[Wiki] Фаза 2/3: обход ссылок (нужно ещё {remaining})...")
        
        # Собираем ссылки из seed-статей
        crawl_seeds = list(SEED_TOPICS[:100])  # Берём первые 100 для crawl
        all_linked = []
        
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_links_from, t, 30): t for t in crawl_seeds}
            for future in as_completed(futures):
                links = future.result()
                for link in links:
                    if link not in titles_done and link not in all_linked:
                        all_linked.append(link)
        
        print(f"  Найдено {len(all_linked)} связанных статей, скачиваю...")
        
        # Скачиваем связанные статьи
        batch_size = min(len(all_linked), remaining + 500)
        to_fetch = all_linked[:batch_size]
        
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_wiki_article, t): t for t in to_fetch}
            done = 0
            for future in as_completed(futures):
                title, text = future.result()
                done += 1
                if text and len(text) > 300 and title not in titles_done:
                    articles.append(text)
                    titles_done.add(title)
                if done % 100 == 0:
                    print(f"  [{done}/{len(to_fetch)}] → {len(articles)} статей")
                if len(articles) >= count:
                    break
        
        print(f"  Фаза 2 завершена: {len(articles)} статей")
    
    # ═══════════════════════════════════════
    # Фаза 3: Random fill (оставшиеся)
    # ═══════════════════════════════════════
    if len(articles) < count:
        remaining = count - len(articles)
        print(f"\n[Wiki] Фаза 3/3: {remaining} случайных статей...")
        
        attempts = 0
        max_attempts = 40
        while len(articles) < count and attempts < max_attempts:
            attempts += 1
            titles = fetch_random_titles(500)
            titles = [t for t in titles if t not in titles_done]
            
            if not titles:
                continue
            
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(fetch_wiki_article, t): t for t in titles}
                for future in as_completed(futures):
                    title, text = future.result()
                    if text and len(text) > 300 and title not in titles_done:
                        articles.append(text)
                        titles_done.add(title)
                    if len(articles) >= count:
                        break
            
            print(f"  [{len(articles)}/{count}] статей")
        
        print(f"  Фаза 3 завершена: {len(articles)} статей")
    
    # ═══════════════════════════════════════
    # Сохранение
    # ═══════════════════════════════════════
    corpus = "\n\n".join(articles)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(corpus)
    
    size_mb = len(corpus.encode('utf-8')) / (1024 * 1024)
    print(f"\n[Wiki] ═══ Готово ═══")
    print(f"  Статей:  {len(articles)}")
    print(f"  Размер:  {size_mb:.1f} MB")
    print(f"  Файл:    {output_path}")
    
    return corpus


def main():
    parser = argparse.ArgumentParser(description="Скачать русскую Википедию для обучения ТАРС")
    parser.add_argument("--count", type=int, default=5000, help="Количество статей (default: 5000)")
    parser.add_argument("--output", type=str, default=None, help="Путь для сохранения")
    args = parser.parse_args()
    
    download_corpus(count=args.count, output_path=args.output)


if __name__ == "__main__":
    main()
