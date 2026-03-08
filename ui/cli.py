"""
═══════════════════════════════════════════════════════════════
  TARS CLI — Interactive Console (Agent 5)
═══════════════════════════════════════════════════════════════

Refactored from launch_tars.py:run_cli().
Uses TarsOrchestrator for the full pipeline.

Owner: Agent 5 (EXCLUSIVE)
"""

import os
import sys
import time
import asyncio
import logging

logger = logging.getLogger("Tars.CLI")

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


async def _run_cli_async(workspace: str = ".", verbose: bool = True):
    """Async CLI main loop."""
    from agent.orchestrator import TarsOrchestrator

    print("\n" + "=" * 60)
    print("   TARS v3.0 — Interactive Console")
    print("   Введите запрос или 'выход' для завершения")
    print("=" * 60 + "\n")

    try:
        orch = TarsOrchestrator(workspace=workspace, verbose=verbose)
    except Exception as e:
        print(f"  ❌ Ошибка инициализации: {e}")
        return

    status = orch.status()
    print(f"  Engine:  {status['engine']}")
    print(f"  Tools:   {', '.join(status['tools'])}")
    print()

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

        # Process
        result = await orch.process(user_input)

        # Display
        print(f"\n  💬 TARS: {result.response}")
        print(f"  📊 [{result.engine_used}] {result.total_time_ms:.0f}ms, "
              f"{result.tokens_generated} tok, mode={result.mode}")

        if result.doubt_scores:
            scores = " ".join(f"{k}={v:.2f}" for k, v in result.doubt_scores.items())
            print(f"  🔍 Doubt: {scores}")

        if result.safety_verdict != "pass":
            print(f"  ⚠️  Safety: {result.safety_verdict}")
        print()


def run_cli(workspace: str = ".", verbose: bool = True):
    """Synchronous CLI entry point."""
    try:
        asyncio.run(_run_cli_async(workspace, verbose))
    except KeyboardInterrupt:
        print("\nTARS: До связи.")


if __name__ == "__main__":
    run_cli()
