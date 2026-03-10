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
    """Проверяет и устанавливает недостающие pip-зависимости из requirements.txt."""
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
    # Use requirements.txt for pinned versions
    req_file = os.path.join(ROOT, "requirements.txt")
    if os.path.exists(req_file):
        logger.info(f"   pip install -r requirements.txt (version-pinned)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", req_file, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=300,
            )
            logger.info("   ✅ Все зависимости установлены")
        except subprocess.TimeoutExpired:
            logger.error("   ❌ Установка превысила таймаут 5 мин")
            return False
        except Exception as e:
            logger.error(f"   ❌ pip install: {e}")
            return False
    else:
        # Fallback: install individually
        for imp_name, pkg_name in missing:
            logger.info(f"   pip install {pkg_name}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_name, "-q"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=120,
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
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=60,
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
        from brain.reflexes.reflex_classifier import ReflexClassifier
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
        from brain.mamba2.mole_router import MoLELayer
        results["MoLE"] = "✅ 8 experts (MoLELayer)"
    except Exception as e:
        results["MoLE"] = f"⚠️  {e}"

    # 8. RAG (Web Search)
    try:
        from tools.web_search import search_duckduckgo
        results["RAG Search"] = "✅ DuckDuckGo"
    except Exception as e:
        results["RAG Search"] = f"❌ {e}"

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
# ═══════════════════════════════════════
# Токенизатор — ЕДИНЫЙ для всех моделей
# ═══════════════════════════════════════
# Убран CharTokenizer(vocab=145) — заменён на cp1251 ByteTokenizer(vocab=256)
from brain.tokenizer import TarsTokenizer


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

        # Токенизатор — cp1251 byte-level (vocab=256)
        tokenizer = TarsTokenizer()
        print(f"  Tokenizer: {tokenizer}")

        # Модель — из config.json + загрузка обученных весов
        model, checkpoint = TarsMamba2LM.load_pretrained(device=device)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        
        if checkpoint:
            print(f"  Model: Mamba-2 LM ({total_params:,} params)")
            print(f"  Weights: {checkpoint}")
        else:
            print(f"  Model: Mamba-2 LM ({total_params:,} params, UNTRAINED)")
            print(f"  ⚠️  Обучите: python training/train_mamba2.py")

        core = get_omega_core()
        print(f"  OmegaCore: {'C++ DLL' if core.available else 'Python fallback'}")
        
        # ═══ Reflex Dispatcher (Спинной мозг) ═══
        dispatcher = ReflexDispatcher(memory=None, max_workers=6)
        print(f"  Reflexes: {len(dispatcher.sensors)} sensors (parallel ThreadPool)")
        print()

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
            waves = stats.get('waves', 0)
            max_waves = stats.get('total_blocks', 12) // 2
            print(f"  │ Depth:         {waves}/{max_waves} waves (2 blocks + merge + spine each)")
            r2 = stats.get('r_squared', 0)
            print(f"  │ p-convergence: {stats.get('final_p', 0):.4f}  (порог: {stats.get('p_threshold', 1.2):.1f}, R²={r2:.2f})")
            converged = stats.get('converged', False)
            if converged:
                print(f"  │ Converged:     ✅ Да")
            else:
                reason = "R²<0.85" if r2 < 0.85 else f"p<{stats.get('p_threshold', 1.2):.1f}"
                print(f"  │ Converged:     ❌ Нет ({reason})")
            print(f"  │ IDME Rounds:   {stats.get('expansion_rounds', 0)}")
            bt = stats.get('branches_tested', 0)
            bw = stats.get('branches_won', 0)
            print(f"  │ Branches:      {bw}/{bt} побед  (3 кандидата/раунд)")
            print(f"  │ Matrices Used: {stats.get('total_matrices', 0)} ({stats.get('matrices_recruited', 0)} рекрутировано)")
            rwkv_mb = stats.get('rwkv_state_size_mb', 0)
            print(f"  │ RWKV State:    {rwkv_mb:.2f} MB (O(1) memory)")
            print(f"  │ Hankel Collapses: {stats.get('hankel_collapses', 0)}")

            # Эксперты — полный цикл волны: experts → merge → spine
            per_wave = stats.get('per_wave_experts', [])
            if per_wave:
                print(f"  │ Waves ({len(per_wave)}):")
                for we in per_wave:
                    wave_num = we.get("wave", "?")
                    # 1) Experts: merge left+right → top-2
                    all_experts = we.get("left", []) + we.get("right", [])
                    import re as _re
                    expert_weights = {}
                    for e in all_experts:
                        m = _re.match(r'(\w+)\((\d+)%\)', e)
                        if m:
                            name, pct = m.group(1), int(m.group(2))
                            expert_weights[name] = expert_weights.get(name, 0) + pct
                    sorted_exp = sorted(expert_weights.items(), key=lambda kv: -kv[1])[:2]
                    total_wt = sum(wt for _, wt in sorted_exp) or 1
                    exp_str = " + ".join(f"{n}({wt*100//total_wt}%)" for n, wt in sorted_exp)
                    # 2) Merge gate alpha
                    alpha = we.get("merge_alpha", 0.5)
                    balance = f"L{(1-alpha)*100:.0f}/R{alpha*100:.0f}"
                    # 3) Spine
                    spine = "✅" if we.get("spine_updated", False) else "—"
                    print(f"  │   W{wave_num}: {exp_str} → merge({balance}) → spine({spine})")
            else:
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
    parser.add_argument("--mode", choices=["cli", "bridge"], default="cli",
                        help="Режим: cli (интерактивный) или bridge (JSON stdin/stdout для Blook)")
    parser.add_argument("--model", type=str, default=None, help="Путь к checkpoint модели")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Макс токенов для генерации")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--format", type=str, default="text", help="Формат вывода")
    args = parser.parse_args()

    # Phase 0: Auto-install
    if not args.no_install:
        if not auto_install():
            logger.error("Установка зависимостей не удалась. Запустите вручную:")
            logger.error("  pip install torch numpy einops tqdm")
            sys.exit(1)

    # Phase 1: Verify
    ok = verify()

    # Phase 2: Launch
    if args.mode == "bridge":
        # Bridge mode for Blook integration (JSON stdin/stdout)
        from tars_bridge_server import TarsBridgeServer
        server = TarsBridgeServer(model_path=args.model, device="auto")
        server.run()
    elif not args.check and ok:
        run_cli()
