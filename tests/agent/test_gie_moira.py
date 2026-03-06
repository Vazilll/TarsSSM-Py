"""GIE agent input validation + MoIRA error boundary tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestGIEAgent:
    """GIE input validation and conversation cap."""

    def test_max_query_len(self):
        from agent.gie import GieAgent
        assert hasattr(GieAgent, 'MAX_QUERY_LEN')
        assert 0 < GieAgent.MAX_QUERY_LEN <= 8192

    def test_max_conversation(self):
        from agent.gie import GieAgent
        assert hasattr(GieAgent, 'MAX_CONVERSATION')
        assert 0 < GieAgent.MAX_CONVERSATION <= 100

    def test_no_getattr_catchall(self):
        from agent.gie import GieAgent
        assert '__getattr__' not in GieAgent.__dict__


class TestMoIRA:
    """MoIRA error boundary tests."""

    def test_has_error_boundary(self):
        from agent.moira import MoIRA
        assert hasattr(MoIRA, '_route_impl')
