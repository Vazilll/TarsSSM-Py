"""T13: Environment variable whitelist and blocked dunders."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestEnvWhitelist:
    """Environment variable whitelist for sandboxed processes."""

    @pytest.fixture
    def engine(self):
        from agent.executor import ActionEngine
        return ActionEngine()

    def test_env_whitelist(self, engine):
        env = engine._safe_env()
        for key in env:
            assert key in engine.SAFE_ENV_VARS or key == 'PYTHONDONTWRITEBYTECODE'

    def test_blocked_dunders_comprehensive(self):
        from agent.executor import ActionEngine
        critical = {'__class__', '__bases__', '__subclasses__', '__globals__', '__builtins__'}
        for dunder in critical:
            assert dunder in ActionEngine._BLOCKED_DUNDERS

    def test_safe_builtins_no_attack_primitives(self):
        from agent.executor import ActionEngine
        dangerous = {'getattr', 'hasattr', 'type', 'isinstance'}
        for name in dangerous:
            assert name not in ActionEngine.SAFE_BUILTINS
