"""ContentCache LRU tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestContentCache:
    """ContentCache O(1) LRU tests."""

    def test_basic(self):
        from tools.sub_agents import ContentCache
        cache = ContentCache(max_size=3)
        assert cache.get("url1") is None and cache.misses == 1
        cache.put("url1", "content1")
        assert cache.get("url1") == "content1" and cache.hits == 1
        cache.put("url2", "c2"); cache.put("url3", "c3"); cache.put("url4", "c4")
        assert cache.get("url1") is None  # evicted

    def test_lru_order(self):
        from tools.sub_agents import ContentCache
        cache = ContentCache(max_size=2)
        cache.put("a", "1"); cache.put("b", "2")
        cache.get("a")        # make 'a' recent
        cache.put("c", "3")   # evict 'b', not 'a'
        assert cache.get("a") == "1" and cache.get("b") is None
