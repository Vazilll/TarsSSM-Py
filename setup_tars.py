import os
import shutil
import logging
import sys
import subprocess

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Tars.Setup")

# Конфигурация путей (Source -> Destination)
SOURCE_BASE = r"C:\Users\Public\Tarsfull"
RUST_PROJECT = os.path.join(SOURCE_BASE, "TarsSSM(rs)")
TARGET_DIR = os.getcwd()

MODEL_MAPPING = {
    # LLM (Mamba-2 / BitNet)
    os.path.join(RUST_PROJECT, r"models\nlp\llm\mamba2-2.7b-q5_k_m.gguf"): "models/llm/mamba2.gguf",
    os.path.join(SOURCE_BASE, r"tars\models\nlp\llm\bitnet-moe-i2_s.gguf"): "models/llm/mamba2_bitnet.gguf",
    os.path.join(RUST_PROJECT, r"models\nlp\llm\tokenizer.json"): "models/llm/tokenizer.json",
    os.path.join(SOURCE_BASE, r"tars\models\nlp\llm\bitnet-moe-tokenizer.json"): "models/llm/bitnet_tokenizer.json",
    
    # Vision (YOLO)
    os.path.join(SOURCE_BASE, r"tars\models\vision\yolo\yolo26n.pt"): "models/vision/yolo26n.pt",
    
    # Voice (Whisper STT, Piper TTS & VAD)
    os.path.join(SOURCE_BASE, r"TarsUltra\models\sherpa-onnx-whisper-tiny\tiny-encoder.onnx"): "models/voice/whisper/tiny-encoder.onnx",
    os.path.join(SOURCE_BASE, r"TarsUltra\models\sherpa-onnx-whisper-tiny\tiny-decoder.onnx"): "models/voice/whisper/tiny-decoder.onnx",
    os.path.join(RUST_PROJECT, r"models\speech\tts\ru_RU-dmitri-medium.onnx"): "models/voice/voice.onnx",
    os.path.join(RUST_PROJECT, r"models\speech\tts\ru_RU-dmitri-medium.onnx.json"): "models/voice/voice.onnx.json",
    os.path.join(RUST_PROJECT, r"models\speech\vad\silero_vad.onnx"): "models/voice/silero_vad.onnx",
    
    # Embeddings (MiniLM)
    os.path.join(RUST_PROJECT, r"models\embeddings\all-MiniLM-L6-v2.onnx"): "models/embeddings/model.onnx",
    os.path.join(RUST_PROJECT, r"models\embeddings\tokenizer.json"): "models/embeddings/tokenizer.json"
}

def setup_directories():
    """ Создание структуры папок проекта. """
    dirs = ["models/llm", "models/vision", "models/voice", "models/embeddings", "data", "hub/static"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Структура папок создана.")

def migrate_models():
    """ Перенос моделей. """
    logger.info("Начинаю перенос моделей...")
    
    # 1. Пофайловый перенос основных моделей
    for src, dst in MODEL_MAPPING.items():
        dst_path = os.path.join(TARGET_DIR, dst)
        if os.path.exists(src):
            try:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src, dst_path)
                logger.info(f"Успешно: {os.path.basename(src)} -> {dst}")
            except Exception as e:
                logger.error(f"Ошибка при копировании {src}: {e}")
                
    # 2. Перенос всей папки эмбеддингов для SentenceTransformer
    emb_src = os.path.join(RUST_PROJECT, r"models\embeddings")
    emb_dst = os.path.join(TARGET_DIR, "models/embeddings")
    if os.path.exists(emb_src):
        try:
            if os.path.exists(emb_dst): shutil.rmtree(emb_dst)
            shutil.copytree(emb_src, emb_dst)
            logger.info("Папка эмбеддингов перенесена полностью.")
        except Exception as e:
            logger.error(f"Ошибка при копировании папки эмбеддингов: {e}")

def fix_nested_structure():
    """ 
    Проверка на лишнюю вложенность (как на скриншоте пользователя).
    Если мы находимся в brain_prototype/brain_prototype, предлагаем подняться выше.
    """
    cwd = os.getcwd()
    if "brain_prototype\\brain_prototype" in cwd:
        logger.warning("Обнаружена лишняя вложенность папок (brain_prototype/brain_prototype).")
        logger.info("Рекомендуется запускать все скрипты из корня: C:\\Users\\Public\\Tarsfull\\TarsSSM(py)")

def install_dependencies():
    """ Установка недостающих библиотек. """
    # Словарь {импорт: имя_пакета_в_pip}
    required = {
        "torch": "torch",
        "ultralytics": "ultralytics",
        "bettercam": "bettercam",
        "faster_whisper": "faster-whisper",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pyautogui": "pyautogui",
        "pygetwindow": "pygetwindow",
        "pyperclip": "pyperclip",
        "mem0": "mem0ai",
        "sentence_transformers": "sentence-transformers"
    }
    
    logger.info("Проверка зависимостей...")
    for imp_name, pkg_name in required.items():
        try:
            __import__(imp_name)
        except ImportError:
            logger.info(f"Установка {pkg_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            except Exception as e:
                logger.error(f"Не удалось установить {pkg_name}: {e}")

if __name__ == "__main__":
    logger.info("=== TARS Python Setup ===")
    fix_nested_structure()
    setup_directories()
    install_dependencies()
    migrate_models()
    logger.info("Настройка завершена. Все системы готовы.")
