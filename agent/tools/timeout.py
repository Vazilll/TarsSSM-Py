"""
═══════════════════════════════════════════════════════════════
  Timeout — Universal Async Timeout Decorator (Agent 5)
═══════════════════════════════════════════════════════════════

Wraps any async tool execution with a configurable timeout.

Owner: Agent 5 (EXCLUSIVE)
"""

import asyncio
import functools
import logging
from typing import Any

logger = logging.getLogger("Tars.Timeout")

DEFAULT_TIMEOUT = 30  # seconds


async def with_timeout(coro, timeout: float = DEFAULT_TIMEOUT, label: str = ""):
    """
    Run an async coroutine with a timeout.

    Args:
        coro: awaitable coroutine
        timeout: max seconds to wait
        label: descriptive label for error messages

    Returns:
        Result of coroutine

    Raises:
        TimeoutError: if coroutine doesn't complete in time
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        msg = f"Timeout ({timeout}s) exceeded"
        if label:
            msg += f" for {label}"
        logger.warning(msg)
        raise TimeoutError(msg)


def timeout_decorator(timeout: float = DEFAULT_TIMEOUT):
    """
    Decorator version of with_timeout.

    Usage:
        @timeout_decorator(10)
        async def my_tool_execute(self, args):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_timeout(
                func(*args, **kwargs),
                timeout=timeout,
                label=func.__qualname__,
            )
        return wrapper
    return decorator
