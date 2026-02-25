"""
launch_tars.py — TARS v3 Auto-Setup & Verification.

1. Автоматически ставит все pip-зависимости
2. Проверяет загрузку каждого модуля
3. Запускает CLI при успехе

Usage:
    python launch_tars.py          # Setup + Verify + CLI
    python launch_tars.py --check  # Только проверка (без CLI)
"""
import sys
import os
import subprocess
import logging
import argparse
import time

# Корень проекта
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Tars.Launcher")

# ═══════════════════════════════════════════
# Phase 0: Auto-Install Dependencies
# ═══════════════════════════════════════════

REQUIRED_PACKAGES = {
    # import_name: pip_package_name
    "torch": "torch",
    "numpy": "numpy",
    "einops": "einops",
    "tqdm": "tqdm",
}

# Опциональные (не блокируют запуск)
OPTIONAL_PACKAGES = {
    "sentencepiece": "sentencepiece",
    "tokenizers": "tokenizers",
    "sounddevice": "sounddevice",
    "duckduckgo_search": "duckduckgo-search",
}


def auto_install():
    """Проверяет и устанавливает недостающие pip-пакеты."""
    missing = []
    for imp_name, pkg_name in REQUIRED_PACKAGES.items():
        try:
            __import__(imp_name)
        except ImportError:
            missing.append((imp_name, pkg_name))

    if not missing:
        logger.info("✅ Все обязательные зависимости установлены")
        return True

    logger.info(f"📦 Установка {len(missing)} недостающих пакетов...")
    for imp_name, pkg_name in missing:
        logger.info(f"   pip install {pkg_name}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg_name, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"   ✅ {pkg_name}")
        except Exception as e:
            logger.error(f"   ❌ {pkg_name}: {e}")
            return False

    # Опциональные — ставим молча, не блокируем
    for imp_name, pkg_name in OPTIONAL_PACKAGES.items():
        try:
            __import__(imp_name)
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_name, "-q"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                pass  # Не критично

    return True


# ═══════════════════════════════════════════
# Phase 1: Verify All Modules
# ═══════════════════════════════════════════

def verify():
    """Проверяет загрузку всех ключевых модулей."""
    results = {}

    # 1. OmegaCore C++ Kernel
    try:
        from brain.omega_core import get_omega_core
        core = get_omega_core()
        if core.available:
            results["OmegaCore"] = f"✅ C++ v{core.version}"
        else:
            results["OmegaCore"] = "⚠️  Python fallback (DLL не скомпилирован)"
    except Exception as e:
        results["OmegaCore"] = f"⚠️  {e}"

    # 2. MinGRU (Tier 1)
    try:
        from brain.min_gru.mingru import MinGRU
        results["MinGRU"] = "✅ Tier 1 Reflex"
    except Exception as e:
        results["MinGRU"] = f"❌ {e}"

    # 3. Reflex Classifier
    try:
        from brain.reflex_classifier import ReflexClassifier
        rc = ReflexClassifier(vocab_size=256, embed_dim=64, hidden_dim=64)
        results["ReflexClassifier"] = f"✅ {rc.count_parameters():,} params"
    except Exception as e:
        results["ReflexClassifier"] = f"❌ {e}"

    # 4. Mamba-2 (Tier 2)
    try:
        from brain.mamba2.model import TarsMamba2LM
        m = TarsMamba2LM(d_model=128, n_layers=2, vocab_size=256, mingru_dim=64)
        info = m.count_parameters()
        total = info["total"] if isinstance(info, dict) else info
        results["Mamba-2 LM"] = f"✅ {total:,} params"
    except Exception as e:
        results["Mamba-2 LM"] = f"❌ {e}"

    # 5. Generator
    try:
        from brain.mamba2.generate_mamba import TarsGenerator
        results["Generator"] = "✅ <thought>/<tool> parser"
    except Exception as e:
        results["Generator"] = f"❌ {e}"

    # 6. RRN
    try:
        from brain.rrn import RrnCore
        results["RRN"] = "✅ Tier 1.5 Relational"
    except Exception as e:
        results["RRN"] = f"⚠️  {e}"

    # 7. MoLE
    try:
        from brain.mole import MoleManager
        results["MoLE"] = "✅ 8 experts"
    except Exception as e:
        results["MoLE"] = f"⚠️  {e}"

    # 8. Knowledge Injector (RAG)
    try:
        from agent.knowledge_injector import KnowledgeInjector
        results["RAG Injector"] = "✅ web/file/recall"
    except Exception as e:
        results["RAG Injector"] = f"❌ {e}"

    # Print results
    print("\n" + "=" * 55)
    print("   TARS v3.0 — System Verification")
    print("=" * 55)
    all_ok = True
    for name, status in results.items():
        icon = "│"
        print(f"  {icon} {name:20s} {status}")
        if "❌" in status:
            all_ok = False
    print("=" * 55)

    if all_ok:
        print("  🎯 ALL CORE SYSTEMS OPERATIONAL\n")
    else:
        print("  ⚠️  Some modules failed\n")

    return all_ok


# ═══════════════════════════════════════════
# Phase 2: Launch CLI
# ═══════════════════════════════════════════

# Русский алфавит + символы (для читаемого вывода даже без обучения)
CHARSET = (
    " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789.,!?;:()-—\"'«»\n"
)

class CharTokenizer:
    """
    Посимвольный токенизатор (char-level).
    Каждый символ = 1 токен. Vocab = len(charset) + 1.
    """
    def __init__(self, charset=CHARSET):
        self.char2id = {ch: i + 1 for i, ch in enumerate(charset)}
        self.id2char = {i + 1: ch for i, ch in enumerate(charset)}
        self.eos_token_id = 0
        self.vocab_size = len(charset) + 1  # +1 для EOS/PAD (id=0)

    def encode(self, text: str) -> list:
        return [self.char2id.get(ch, 1) for ch in text]

    def decode(self, ids: list) -> str:
        return "".join(self.id2char.get(i, "") for i in ids if i != 0)


def run_cli():
    """Запускает интерактивный CLI с подробным отображением метрик мозга."""
    print("\n" + "=" * 60)
    print("   TARS v3.0 — Interactive Console (Verbose Brain Mode)")
    print("   Введите запрос или 'выход' для завершения")
    print("=" * 60 + "\n")

    try:
        from brain.mamba2.model import TarsMamba2LM
        from brain.mamba2.generate_mamba import TarsGenerator, GenerationConfig
        from brain.omega_core import get_omega_core
        from brain.reflexes.reflex_dispatcher import ReflexDispatcher
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        # Токенизатор
        tokenizer = CharTokenizer()
        print(f"  Tokenizer: CharTokenizer (vocab={tokenizer.vocab_size}, chars={tokenizer.vocab_size - 1})")

        # Модель — vocab_size ТОЧНО совпадает с токенизатором
        model = TarsMamba2LM(
            d_model=256, n_layers=4, vocab_size=tokenizer.vocab_size, mingru_dim=128
        ).to(device)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: Mamba-2 LM ({total_params:,} params, UNTRAINED)")

        core = get_omega_core()
        print(f"  OmegaCore: {'C++ DLL' if core.available else 'Python fallback'}")
        
        # ═══ Reflex Dispatcher (Спинной мозг) ═══
        dispatcher = ReflexDispatcher(memory=None, max_workers=6)
        print(f"  Reflexes: {len(dispatcher.sensors)} sensors (parallel ThreadPool)")
        
        print(f"\n  ⚠️  Модель НЕ обучена — вывод случайный!")
        print(f"  Обучите: python training/train_mamba2.py\n")

        gen = TarsGenerator(model, tokenizer, omega_core=core)
        config = GenerationConfig(max_tokens=64, temperature=0.9, top_k=40, top_p=0.92)

        while True:
            try:
                user_input = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ["выход", "exit", "quit", "стоп"]:
                print("\nTARS: До связи.")
                break

            # ═══════════════════════════════════════
            # STEP 1: Reflex Dispatch (6 sensors × parallel)
            # ═══════════════════════════════════════
            reflex_ctx = dispatcher.dispatch(user_input)
            
            print(f"\n{'─' * 60}")
            print(f"  ⚡ ═══ Reflexes ({reflex_ctx.dispatch_time_ms:.0f}ms) ═══")
            print(f"  │ {reflex_ctx.summary_line()}")
            
            # Sensor timing breakdown
            timings = " ".join(
                f"{name}:{ms:.0f}ms"
                for name, ms in sorted(reflex_ctx.sensor_times.items(), key=lambda x: -x[1])
            )
            print(f"  │ Sensors: {timings}")
            
            if reflex_ctx.dominant_emotion != "neutral":
                print(f"  │ Emotion: {reflex_ctx.dominant_emotion} (urgency={reflex_ctx.urgency:.0%})")
            print(f"  │ System:  CPU {reflex_ctx.cpu_percent:.0f}%, RAM {reflex_ctx.ram_free_gb:.1f}GB, GPU {'✅' if reflex_ctx.gpu_available else '❌'}")
            if reflex_ctx.is_followup:
                print(f"  │ Context: ↩️ Follow-up (session #{reflex_ctx.session_length})")
            if reflex_ctx.rag_found:
                print(f"  │ RAG:     {len(reflex_ctx.rag_snippets)} docs found")
            print(f"  ╰{'─' * 45}")
            
            # ═══ Fast response (no brain needed) ═══
            if reflex_ctx.can_handle_fast and reflex_ctx.fast_response:
                print(f"\n  💬 TARS (рефлекс): {reflex_ctx.fast_response}")
                dispatcher.add_to_history(user_input, reflex_ctx.fast_response, reflex_ctx.intent)
                print(f"\n  📊 Reflex handled ({reflex_ctx.dispatch_time_ms:.0f}ms, brain NOT invoked)")
                print(f"{'─' * 60}")
                continue

            # ═══════════════════════════════════════
            # STEP 2: Full Brain Think (with enriched context)
            # ═══════════════════════════════════════
            input_ids = tokenizer.encode(user_input)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            print(f"\n  📝 Input: \"{user_input}\"")
            print(f"  🔢 Tokens: {input_ids[:20]}{'...' if len(input_ids) > 20 else ''} (len={len(input_ids)})")

            # ── Think (IDME deep reasoning + adaptive depth) ──
            t0 = time.time()
            think_result = model.think(
                input_tensor,
                query_text=user_input,
                reflex_ctx=reflex_ctx,
            )

            if isinstance(think_result, tuple):
                logits, stats = think_result
            else:
                logits = think_result
                stats = {}

            think_time = (time.time() - t0) * 1000

            # ── Отображение метрик мозга ──
            print(f"\n  🧠 ═══ Brain Think (Deep WuNeng Core: Mamba-2 + RWKV-7) ═══")
            print(f"  │ Task Type:     {stats.get('task_type', '?')}")
            be = stats.get('blocks_executed', '?')
            ed = stats.get('estimated_depth', '?')
            tb = stats.get('total_blocks', '?')
            print(f"  │ Depth:         {be}/{tb} blocks (target: {ed})")
            print(f"  │ p-convergence: {stats.get('final_p', 0):.4f}  (порог: {stats.get('p_threshold', 1.2):.1f})")
            print(f"  │ Converged:     {'✅ Да' if stats.get('converged', False) else '❌ Нет'}")
            print(f"  │ IDME Rounds:   {stats.get('expansion_rounds', 0)}")
            bt = stats.get('branches_tested', 0)
            bw = stats.get('branches_won', 0)
            print(f"  │ Branches:      {bw}/{bt} побед  (3 кандидата/раунд)")
            print(f"  │ Matrices Used: {stats.get('total_matrices', 0)} ({stats.get('matrices_recruited', 0)} рекрутировано)")
            rwkv_mb = stats.get('rwkv_state_size_mb', 0)
            print(f"  │ RWKV State:    {rwkv_mb:.2f} MB (O(1) memory)")
            print(f"  │ Hankel Collapses: {stats.get('hankel_collapses', 0)}")

            # Эксперты
            experts = stats.get('active_experts', [])
            if experts:
                expert_str = ", ".join(experts) if isinstance(experts[0], str) else str(experts)
                print(f"  │ MoLE Experts:  {expert_str}")

            print(f"  │ Think Time:    {stats.get('total_ms', think_time):.0f}ms")
            print(f"  │ Logits Shape:  {list(logits.shape)}")

            # Top-5 предсказаний
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top5_probs, top5_ids = probs.topk(5)
            print(f"  │")
            print(f"  │ Top-5 Next Tokens:")
            for prob, tid in zip(top5_probs, top5_ids):
                char = tokenizer.decode([tid.item()])
                char_display = repr(char) if char in ['\n', ' '] else char
                print(f"  │   '{char_display}' (id={tid.item():3d})  p={prob.item():.4f}")

            print(f"  ╰{'─' * 45}")

            # ── Generate ──
            print(f"\n  💬 Генерация:")
            print(f"  ", end="")

            t1 = time.time()
            result = gen.generate(
                user_input, config=config,
                on_token=lambda t: print(t, end="", flush=True)
            )
            gen_time = time.time() - t1

            # ── Итоговая статистика ──
            tps = result.tokens_generated / gen_time if gen_time > 0 else 0
            print(f"\n\n  📊 ═══ Generation Stats ═══")
            print(f"  │ Tokens:     {result.tokens_generated}")
            print(f"  │ Time:       {gen_time:.2f}s ({tps:.1f} tok/s)")
            print(f"  │ p-final:    {result.p_convergence:.4f}")
            print(f"  │ IDME rounds:{result.idme_rounds}")
            if result.tool_calls:
                print(f"  │ Tool calls: {result.tool_calls}")
            if result.thought:
                print(f"  │ Thought:    {result.thought[:80]}...")
            print(f"  ╰{'─' * 45}\n")
            
            # Update session history
            dispatcher.add_to_history(user_input, result.text, reflex_ctx.intent)

    except Exception as e:
        logger.error(f"CLI Error: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TARS v3 Launcher")
    parser.add_argument("--check", action="store_true", help="Только проверка (без CLI)")
    parser.add_argument("--no-install", action="store_true", help="Пропустить установку зависимостей")
    args = parser.parse_args()

    # Phase 0: Auto-install
    if not args.no_install:
        if not auto_install():
            logger.error("Установка зависимостей не удалась. Запустите вручную:")
            logger.error("  pip install torch numpy einops tqdm")
            sys.exit(1)

    # Phase 1: Verify
    ok = verify()

    # Phase 2: CLI (если всё ок и не --check)
    if not args.check and ok:
        run_cli()
