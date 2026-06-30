"""Unit tests for the LLM router strategy module (strategies/ package).

All scoring tests pass alpha, load_threshold, kv_cache_usage_perc, and tier
weights explicitly so that the inline calculation comments match the code
without requiring knowledge of default values.

Load formula: S_load = (1 - kv_usage) / (1 + running + waiting)  ∈ [0, 1]
Overloaded when S_load < load_threshold (default 0.1).
Overloaded replica: cache zeroed, score = (1-alpha) × S_load.
Healthy replica:    score = alpha × S_cache + (1-alpha) × S_load.
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.strategies import (
    KVCacheAwareStrategy,
    RoutingStrategy,
    StrategyError,
    StrategyRegistry,
    route,
)
from uni_agent.llm_router.strategies.base import ReplicaInfo
from uni_agent.llm_router.collectors.metric_spec import MetricKey


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _strat(**kwargs) -> KVCacheAwareStrategy:
    """Build a KVCacheAwareStrategy with required boilerplate fields filled in."""
    defaults = dict(
        alpha=0.7,
        load_threshold=0.1,
        layer_weights={"cpu": 1.0, "ssd": 0.25},
        collector_names=["vllm_zmq"],
        weight=1.0,
    )
    defaults.update(kwargs)
    return KVCacheAwareStrategy(**defaults)


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


PROMPT_IDS = [1, 2, 3]


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeRouteDataProvider:
    """In-memory replica metrics for unit tests.

    Each replica entry is a plain dict with the following optional keys:
      kv_cache_usage_perc  – KV cache usage ratio (default 1.0)
      num_requests_running – requests in flight (default 0)
      num_requests_waiting – requests in the queue (default 0)
      gpu_hit_pct          – GPU prefix cache hit percent 0-100 (default 0)
      tiers                – dict mapping tier name to hit rate (default {})
    """

    def __init__(self, data: dict[str, dict]):
        self._data = data

    def get_metric(self, replica_id: str, key: str) -> float | int:
        entry = self._data.get(replica_id, {})
        if key == MetricKey.KV_CACHE_USAGE_PERC:
            return entry.get("kv_cache_usage_perc", 1.0)
        if key == MetricKey.NUM_REQUESTS_RUNNING:
            return entry.get("num_requests_running", 0)
        if key == MetricKey.NUM_REQUESTS_WAITING:
            return entry.get("num_requests_waiting", 0)
        return entry.get(key, 0.0)

    def get_metrics(self, replica_id: str) -> dict:
        entry = self._data.get(replica_id, {})
        return {
            MetricKey.KV_CACHE_USAGE_PERC: entry.get("kv_cache_usage_perc", 1.0),
            MetricKey.NUM_REQUESTS_RUNNING: entry.get("num_requests_running", 0),
            MetricKey.NUM_REQUESTS_WAITING: entry.get("num_requests_waiting", 0),
        }

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Returns {replica_id: hit_percent 0-100} for replicas with hits."""
        result = {}
        for replica_id, entry in self._data.items():
            pct = entry.get("gpu_hit_pct", 0)
            if pct > 0:
                result[replica_id] = pct
        return result

    def get_tier_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int], tier: str) -> float:
        return self._data.get(replica_id, {}).get("tiers", {}).get(tier, 0.0)


class ConstantStrategy:
    """Returns a fixed per-replica score list (for route() composition tests)."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score(self, prompt_ids, provider, replicas) -> list[float]:
        return list(self._scores)


class BadLengthStrategy:
    """Returns a wrong-length list to exercise the contract check in route()."""

    def score(self, prompt_ids, provider, replicas) -> list[float]:
        return [1.0]


class RaisingStrategy:
    """Raises inside score() to exercise route()'s exception wrapping."""

    def score(self, prompt_ids, provider, replicas) -> list[float]:
        raise KeyError("boom")


# --------------------------------------------------------------------------- #
# Comprehensive scenario: overload + GPU hit + tier hit
# --------------------------------------------------------------------------- #
class TestKVCAwareComprehensive:
    def test_overload_gpu_hit_tier_hit(self):
        """
        Feature: partial overload with fast path — overloaded replica cannot outrank healthy ones
        Description: three replicas in partial overload scenario:
          - rep_a: healthy, kv=0.3, r=1, w=0 → S_load=0.35; has GPU hit → fast path applies
          - rep_b: healthy, kv=0.5, r=1, w=0 → S_load=0.25; no GPU hit, has tier hit (slow path)
          - rep_c: overloaded, kv=0.9, r=5, w=0 → S_load=0.0167; gpu_hit=90% (ignored, cache zeroed)
          Fast/slow path decision uses only healthy replicas: rep_a has gpu_hit > 0 → fast path.
          rep_b scores by gpu_hit=0 (fast path, tier ignored). rep_c cache is zeroed.
        Expectation: scores are [0.665, 0.075, 0.005]; ranking is rep_a > rep_b > rep_c
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # rep_a: S_load = (1-0.3)/(1+1+0) = 0.7/2 = 0.35  healthy; gpu_hit_pct=80 → fast path
                "rep_a": {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 80, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                # rep_b: S_load = (1-0.5)/(1+1+0) = 0.5/2 = 0.25  healthy; no gpu_hit, tier ignored in fast path
                "rep_b": {"kv_cache_usage_perc": 0.5, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 0.6, "ssd": 0.2}},
                # rep_c: S_load = (1-0.9)/(1+5+0) = 0.1/6 ≈ 0.0167  overloaded; gpu_hit_pct=90 ignored
                "rep_c": {"kv_cache_usage_perc": 0.9, "num_requests_running": 5, "num_requests_waiting": 0,
                          "gpu_hit_pct": 90, "tiers": {"cpu": 0.9, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"))

        # fast path (rep_a healthy with gpu_hit_pct > 0), gpu_hit = 80/100 = 0.8
        # rep_a: 0.7 * 0.8 + 0.3 * 0.35   = 0.560 + 0.105  = 0.665
        # rep_b: 0.7 * 0.0 + 0.3 * 0.25   = 0.000 + 0.075  = 0.075  (gpu_hit=0 in fast path)
        # rep_c: (1-0.7) * (0.1/6)         = 0.3   * 0.0167 ≈ 0.005  (overloaded: cache zeroed)
        assert scores == pytest.approx([0.665, 0.075, 0.005], rel=1e-3)

        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"))
        assert ranking == ["rep_a", "rep_b", "rep_c"]

        # rep_c has highest gpu_hit but lowest score — proves overload cache zeroing works
        assert scores[2] < scores[1] < scores[0]


# --------------------------------------------------------------------------- #
# StrategyRegistry
# --------------------------------------------------------------------------- #
class TestStrategyRegistry:
    def test_builtin_registered(self):
        """
        Feature: built-in KVCacheAwareStrategy is pre-registered for KVCAwareStrategyConfig
        Description: look up KVCAwareStrategyConfig in the registry
        Expectation: returns KVCacheAwareStrategy class
        """
        from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig
        assert StrategyRegistry.get(KVCAwareStrategyConfig) is KVCacheAwareStrategy

    def test_register_and_get(self):
        """
        Feature: custom strategy can be registered and retrieved by config type
        Description: register a dummy config class, then call get() with it
        Expectation: get() returns the registered strategy class
        """
        class _DummyConfig:
            pass

        class _DummyStrategy:
            def score(self, prompt_ids, provider, replicas):
                return [0.0] * len(replicas)

        StrategyRegistry.register(_DummyConfig, _DummyStrategy)
        try:
            assert StrategyRegistry.get(_DummyConfig) is _DummyStrategy
        finally:
            StrategyRegistry._registry.pop(_DummyConfig, None)

    def test_get_unknown_raises(self):
        """
        Feature: looking up an unregistered config type raises KeyError
        Description: call get() with a class that was never registered
        Expectation: raises KeyError
        """
        class _UnknownConfig:
            pass

        with pytest.raises(KeyError):
            StrategyRegistry.get(_UnknownConfig)


# --------------------------------------------------------------------------- #
# S_load formula and overload boundary
# --------------------------------------------------------------------------- #
class TestKVCAwareLoad:
    def test_s_load_formula(self):
        """
        Feature: S_load = (1-kv) / (1+running+waiting), overloaded when S_load < threshold
        Description: three replicas covering idle / near-threshold / overloaded states
        Expectation: idle scores highest; overloaded gets cache-zeroed penalty
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # idle:      S_load = (1-0.0)/(1+0+0) = 1.0   healthy
                "idle":      {"kv_cache_usage_perc": 0.0, "num_requests_running": 0, "num_requests_waiting": 0,
                              "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                # healthy:   S_load = (1-0.88)/(1+0+0) = 0.12  healthy (above threshold=0.1)
                "healthy":   {"kv_cache_usage_perc": 0.88, "num_requests_running": 0, "num_requests_waiting": 0,
                              "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                # overloaded: S_load = (1-0.92)/(1+0+0) = 0.08  overloaded (below threshold=0.1)
                "overloaded": {"kv_cache_usage_perc": 0.92, "num_requests_running": 0, "num_requests_waiting": 0,
                               "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("idle", "healthy", "overloaded"))

        # no gpu_hit anywhere → slow path; no tier hits → S_cache=0
        # idle:       0.7*0 + 0.3*1.0  = 0.300
        # healthy:    0.7*0 + 0.3*0.12 = 0.036
        # overloaded: (1-0.7)*0.08     = 0.024  (cache zeroed)
        assert scores == pytest.approx([0.300, 0.036, 0.024], rel=1e-3)
        assert scores[0] > scores[1] > scores[2]

    def test_waiting_queue_increases_load(self):
        """
        Feature: num_requests_waiting contributes to load alongside num_requests_running
        Description: two replicas with same kv and running count; one has waiting requests
        Expectation: replica with waiting requests has lower S_load and lower score
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # no_wait:   S_load = (1-0.5)/(1+5+0) = 0.5/6 = 0.0833  overloaded
                "no_wait":   {"kv_cache_usage_perc": 0.5, "num_requests_running": 5, "num_requests_waiting": 0,
                              "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                # with_wait: S_load = (1-0.5)/(1+5+5) = 0.5/11 = 0.0455  overloaded (lower)
                "with_wait": {"kv_cache_usage_perc": 0.5, "num_requests_running": 5, "num_requests_waiting": 5,
                              "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("no_wait", "with_wait"))

        # both overloaded (S_load < 0.1) → cache zeroed → score = 0.3 * S_load
        # no_wait:   0.3 * (0.5/6)  = 0.025
        # with_wait: 0.3 * (0.5/11) ≈ 0.01364
        assert scores == pytest.approx([0.025, 0.5 * 0.3 / 11], rel=1e-3)
        assert scores[0] > scores[1]

    def test_kv_usage_scales_load(self):
        """
        Feature: higher kv_cache_usage_perc produces lower S_load
        Description: two replicas with same request counts but different kv usage
        Expectation: higher kv replica has lower S_load and lower score
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # kv_low:  S_load = (1-0.2)/(1+2+0) = 0.8/3 = 0.267  healthy
                "kv_low":  {"kv_cache_usage_perc": 0.2, "num_requests_running": 2, "num_requests_waiting": 0,
                            "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                # kv_high: S_load = (1-0.8)/(1+2+0) = 0.2/3 = 0.067  overloaded
                "kv_high": {"kv_cache_usage_perc": 0.8, "num_requests_running": 2, "num_requests_waiting": 0,
                            "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("kv_low", "kv_high"))

        # kv_low  healthy:    0.7*0 + 0.3*0.267 = 0.080
        # kv_high overloaded: (1-0.7)*0.067     = 0.020
        assert scores == pytest.approx([0.080, 0.020], rel=1e-3)
        assert scores[0] > scores[1]

    def test_missing_metrics_defaults_to_overloaded(self):
        """
        Feature: unknown replica defaults to kv=1.0 (pessimistic), treated as overloaded
        Description: score a replica whose id is not present in the provider
        Expectation: defaults give kv=1.0 → S_load=0 → overloaded → score=0
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider({})
        scores = strat.score(PROMPT_IDS, provider, _replicas("ghost"))
        # S_load = (1-1.0)/(1+0+0) = 0.0 < 0.1 → overloaded; score = 0.3*0 = 0.0
        assert scores == pytest.approx([0.0])

    @pytest.mark.parametrize("kwargs", [
        {"alpha": 1.5},
        {"alpha": -0.1},
        {"load_threshold": 0},
        {"load_threshold": -0.1},
        {"load_threshold": 1.0},
        {"layer_weights": {"cpu": -1.0}},
        {"layer_weights": {"nvme": 1.0}},
        {"layer_weights": {"cpu": 1.0, "nvme": 0.5}},
    ])
    def test_construction_validation(self, kwargs):
        """
        Feature: invalid constructor arguments raise StrategyError
        Description: construct KVCacheAwareStrategy with each invalid kwarg
        Expectation: raises StrategyError for each case
        """
        with pytest.raises(StrategyError):
            _strat(**kwargs)

    def test_alpha_one_overloaded_cannot_outrank_healthy(self):
        """
        Feature: with alpha=1.0, overloaded replica still ranks below healthy
        Description: overloaded replica has high gpu_hit; healthy has gpu_hit=50%; alpha=1.0
        Expectation: overloaded score=0 (cache zeroed, load dropped at alpha=1); healthy score=0.5
        """
        strat = _strat(alpha=1.0, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # overloaded: S_load = (1-0.92)/(1+0+0) = 0.08 < 0.1 → cache zeroed
                "overloaded": {"kv_cache_usage_perc": 0.92, "num_requests_running": 0, "num_requests_waiting": 0,
                               "gpu_hit_pct": 90, "tiers": {"cpu": 1.0, "ssd": 0.0}},
                # healthy: S_load = (1-0.3)/(1+1+0) = 0.35 ≥ 0.1; gpu_hit_pct=50 → fast path
                "healthy":    {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                               "gpu_hit_pct": 50, "tiers": {"cpu": 0.0, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("overloaded", "healthy"))

        # overloaded: (1-1.0)*0.08 = 0.0   (cache zeroed, (1-alpha) term also 0)
        # healthy:    1.0*0.5 + 0.0*0.35  = 0.5
        assert scores == pytest.approx([0.0, 0.5])
        assert scores[1] > scores[0]
        assert route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("overloaded", "healthy")) == ["healthy", "overloaded"]


# --------------------------------------------------------------------------- #
# Fast / slow path selection
# --------------------------------------------------------------------------- #
class TestKVCAwareFastSlow:
    def test_healthy_with_gpu_hit_uses_fast_path(self):
        """
        Feature: fast path activates when any healthy replica has GPU hit > 0
        Description: two healthy replicas; rep_a has gpu_hit_pct=70, rep_b has none; tier hits set
        Expectation: tier hits are ignored; scoring uses gpu_hit values (fast path)
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # rep_a: S_load = (1-0.3)/(1+1+0) = 0.35  healthy; gpu_hit_pct=70 triggers fast path
                "rep_a": {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 70, "tiers": {"cpu": 1.0, "ssd": 0.0}},
                # rep_b: S_load = (1-0.5)/(1+1+0) = 0.25  healthy; no gpu_hit; tier ignored in fast path
                "rep_b": {"kv_cache_usage_perc": 0.5, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 1.0, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))

        # fast path (rep_a has gpu_hit_pct > 0) — tier hits on both replicas are ignored
        # rep_a: 0.7*(70/100) + 0.3*0.35 = 0.490 + 0.105 = 0.595
        # rep_b: 0.7*0.0 + 0.3*0.25 = 0.000 + 0.075 = 0.075
        assert scores == pytest.approx([0.595, 0.075])

    def test_env_can_force_slow_path_when_gpu_hit_exists(self, monkeypatch):
        """
        Feature: verification can force tier slow path even when GPU hits exist
        Description: set UNI_AGENT_FORCE_TIER_SLOW_PATH=1 with a healthy GPU-hit replica
        Expectation: tier hit rates are queried and GPU hit scores are ignored
        """
        class _CountingProvider(FakeRouteDataProvider):
            def __init__(self, data):
                super().__init__(data)
                self.tier_calls: list[tuple[str, str]] = []

            def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
                self.tier_calls.append((replica_id, tier))
                return super().get_tier_prefix_hit_rate(replica_id, prompt_ids, tier)

        monkeypatch.setenv("UNI_AGENT_FORCE_TIER_SLOW_PATH", "1")
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = _CountingProvider(
            {
                # rep_a has GPU hit, but forced slow path should use only SSD tier hit.
                "rep_a": {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 70, "tiers": {"cpu": 0.0, "ssd": 1.0}},
                "rep_b": {"kv_cache_usage_perc": 0.5, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 0.8, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))

        # Forced slow path:
        # rep_a: slow_cache = 0.0*1.0 + 1.0*0.25 = 0.25; score = 0.7*0.25 + 0.3*0.35 = 0.280
        # rep_b: slow_cache = 0.8*1.0 + 0.0*0.25 = 0.80; score = 0.7*0.80 + 0.3*0.25 = 0.635
        assert scores == pytest.approx([0.280, 0.635])
        assert ("rep_a", "ssd") in provider.tier_calls
        assert ("rep_b", "cpu") in provider.tier_calls

    def test_no_gpu_hit_uses_slow_path(self):
        """
        Feature: slow path activates when no healthy replica has GPU hit > 0
        Description: two healthy replicas both with gpu_hit_pct=0; different tier hits
        Expectation: scoring uses tier-weighted cache values
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # rep_a: S_load = (1-0.3)/(1+1+0) = 0.35; slow_cache = 0.6*1.0 + 0.2*0.25 = 0.65
                "rep_a": {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 0.6, "ssd": 0.2}},
                # rep_b: S_load = (1-0.5)/(1+1+0) = 0.25; slow_cache = 0.3*1.0 + 0.4*0.25 = 0.40
                "rep_b": {"kv_cache_usage_perc": 0.5, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 0.3, "ssd": 0.4}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))

        # slow path (no healthy replica has gpu_hit_pct > 0)
        # rep_a: 0.7*0.65 + 0.3*0.35 = 0.455 + 0.105 = 0.560
        # rep_b: 0.7*0.40 + 0.3*0.25 = 0.280 + 0.075 = 0.355
        assert scores == pytest.approx([0.560, 0.355])

    def test_overloaded_replica_excluded_from_path_decision(self):
        """
        Feature: fast/slow path decision only considers healthy replicas
        Description: overloaded replica has gpu_hit_pct=90; the only healthy replica has gpu_hit_pct=0
        Expectation: slow path used (healthy subset has no GPU hit); overloaded replica cache zeroed
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # healthy:    S_load = (1-0.3)/(1+1+0) = 0.35 ≥ 0.1; no gpu_hit → slow path
                "healthy":    {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                               "gpu_hit_pct": 0, "tiers": {"cpu": 0.6, "ssd": 0.0}},
                # overloaded: S_load = (1-0.92)/(1+0+0) = 0.08 < 0.1; gpu_hit_pct=90 ignored
                "overloaded": {"kv_cache_usage_perc": 0.92, "num_requests_running": 0, "num_requests_waiting": 0,
                               "gpu_hit_pct": 90, "tiers": {"cpu": 0.9, "ssd": 0.0}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("healthy", "overloaded"))

        # slow path; healthy: slow_cache = 0.6*1.0 + 0.0*0.25 = 0.6
        # healthy:    0.7*0.6 + 0.3*0.35 = 0.420 + 0.105 = 0.525
        # overloaded: (1-0.7)*0.08        = 0.3*0.08       = 0.024  (cache zeroed)
        assert scores == pytest.approx([0.525, 0.024], rel=1e-3)
        assert scores[0] > scores[1]

    def test_all_overloaded_forces_slow_path(self):
        """
        Feature: all-overloaded state forces slow path; cache + load both contribute
        Description: three replicas all overloaded with different tier hits and S_load values
        Expectation: slow path scores combine tier cache and load; higher gpu_hit does not help
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # rep_a: S_load = (1-0.92)/1 = 0.08; slow_cache = 0.65*1.0 + 0.1*0.25 = 0.675
                "rep_a": {"kv_cache_usage_perc": 0.92, "num_requests_running": 0, "num_requests_waiting": 0,
                          "gpu_hit_pct": 20, "tiers": {"cpu": 0.65, "ssd": 0.1}},
                # rep_b: S_load = (1-0.95)/1 = 0.05; slow_cache = 0.9*1.0 + 0.0*0.25 = 0.9
                "rep_b": {"kv_cache_usage_perc": 0.95, "num_requests_running": 0, "num_requests_waiting": 0,
                          "gpu_hit_pct": 90, "tiers": {"cpu": 0.90, "ssd": 0.0}},
                # rep_c: S_load = (1-0.92)/1 = 0.08; slow_cache = 0.4*1.0 + 0.2*0.25 = 0.45
                "rep_c": {"kv_cache_usage_perc": 0.92, "num_requests_running": 0, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 0.40, "ssd": 0.2}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"))

        # all overloaded → slow path; full formula alpha*S_cache + (1-alpha)*S_load for all
        # rep_a: 0.7*0.675 + 0.3*0.08 = 0.4725 + 0.024 = 0.4965
        # rep_b: 0.7*0.900 + 0.3*0.05 = 0.6300 + 0.015 = 0.6450  (high cache wins despite lower S_load)
        # rep_c: 0.7*0.450 + 0.3*0.08 = 0.3150 + 0.024 = 0.3390
        assert scores == pytest.approx([0.4965, 0.6450, 0.3390], rel=1e-3)
        # rep_b has highest gpu_hit (90%) but gpu_hit is not used here — proves forced slow path
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"))
        assert ranking == ["rep_b", "rep_a", "rep_c"]


# --------------------------------------------------------------------------- #
# Tier weights in slow path
# --------------------------------------------------------------------------- #
class TestKVCAwareTierWeights:
    def test_cpu_tier_weight_higher_than_ssd(self):
        """
        Feature: cpu tier weight (1.0) is higher than ssd tier weight (0.25)
        Description: two replicas with same S_load; rep_a has cpu hit, rep_b has ssd hit
        Expectation: rep_a scores higher because cpu tier contributes more
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                # both: S_load = (1-0.5)/(1+0+0) = 0.5  healthy; no gpu_hit → slow path
                "cpu_hit": {"kv_cache_usage_perc": 0.5, "num_requests_running": 0, "num_requests_waiting": 0,
                            "gpu_hit_pct": 0, "tiers": {"cpu": 0.6, "ssd": 0.0}},
                "ssd_hit": {"kv_cache_usage_perc": 0.5, "num_requests_running": 0, "num_requests_waiting": 0,
                            "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.8}},
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("cpu_hit", "ssd_hit"))

        # slow path; S_load=0.5 for both
        # cpu_hit: slow_cache = 0.6*1.0 + 0.0*0.25 = 0.6; score = 0.7*0.6 + 0.3*0.5 = 0.570
        # ssd_hit: slow_cache = 0.0*1.0 + 0.8*0.25 = 0.2; score = 0.7*0.2 + 0.3*0.5 = 0.290
        assert scores == pytest.approx([0.570, 0.290])
        assert scores[0] > scores[1]

    def test_tier_none_treated_as_zero(self):
        """
        Feature: None return from get_tier_prefix_hit_rate is treated as 0.0
        Description: provider returns None for tier hit rate; slow path active
        Expectation: score equals pure load score (S_cache=0), no TypeError raised
        """
        class _NoneProvider(FakeRouteDataProvider):
            def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
                return None

        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = _NoneProvider(
            {"rep": {"kv_cache_usage_perc": 0.5, "num_requests_running": 0, "num_requests_waiting": 0,
                     "gpu_hit_pct": 0, "tiers": {}}}
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep"))
        # S_load = (1-0.5)/(1+0+0) = 0.5; S_cache = 0 (None → 0.0); score = 0.7*0 + 0.3*0.5 = 0.15
        assert scores == pytest.approx([0.15])


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #
class TestStrategyContract:
    def test_protocol_satisfied(self):
        """
        Feature: KVCacheAwareStrategy satisfies the RoutingStrategy Protocol
        Description: check isinstance against RoutingStrategy (runtime_checkable)
        Expectation: returns True
        """
        strat = _strat()
        assert isinstance(strat, RoutingStrategy)

    def test_output_length_matches_replicas(self):
        """
        Feature: score() returns a list with same length as replicas
        Description: score two replicas and check output length
        Expectation: len(scores) == 2
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 90, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                "rep_b": {"kv_cache_usage_perc": 0.5, "num_requests_running": 2, "num_requests_waiting": 0,
                          "gpu_hit_pct": 10, "tiers": {"cpu": 0.0, "ssd": 0.0}},
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas)
        assert len(scores) == len(replicas)

    def test_stateless_repeatable(self):
        """
        Feature: calling score() twice on the same inputs produces identical results
        Description: call score() twice with the same provider and replica list
        Expectation: both calls return approx-equal results
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 80, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                "rep_b": {"kv_cache_usage_perc": 0.5, "num_requests_running": 2, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 0.5, "ssd": 0.0}},
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat.score(PROMPT_IDS, provider, replicas) == pytest.approx(strat.score(PROMPT_IDS, provider, replicas))


# --------------------------------------------------------------------------- #
# route() composition
# --------------------------------------------------------------------------- #
class TestRoute:
    def test_single_strategy_descending(self):
        """
        Feature: route() returns replica ids sorted by score descending
        Description: single strategy with scores [0.2, 0.5, 0.1]
        Expectation: ranking is ["rep_b", "rep_a", "rep_c"]
        """
        provider = FakeRouteDataProvider({})
        ranking = route(
            [(ConstantStrategy([0.2, 0.5, 0.1]), 1.0)],
            PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"),
        )
        assert ranking == ["rep_b", "rep_a", "rep_c"]

    def test_multi_strategy_weighted_sum(self):
        """
        Feature: multiple strategies are combined by weighted sum
        Description: two strategies with weight=0.5 each
        Expectation: final = [0.5*1+0.5*3, 0.5*2+0.5*1, 0.5*3+0.5*0] = [2.0, 1.5, 1.5] → rep_a first
        """
        provider = FakeRouteDataProvider({})
        strategies = [
            (ConstantStrategy([1.0, 2.0, 3.0]), 0.5),
            (ConstantStrategy([3.0, 1.0, 0.0]), 0.5),
        ]
        ranking = route(strategies, PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"))
        assert ranking[0] == "rep_a"

    def test_overloaded_present_in_ranking(self):
        """
        Feature: overloaded replicas remain in ranking as fallback
        Description: two healthy replicas and one overloaded; all three should appear in ranking
        Expectation: overloaded replica ranked last; all three ids present
        """
        strat = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                "rep_a":     {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                              "gpu_hit_pct": 80, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                "rep_b":     {"kv_cache_usage_perc": 0.5, "num_requests_running": 1, "num_requests_waiting": 0,
                              "gpu_hit_pct": 30, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                "overloaded": {"kv_cache_usage_perc": 0.92, "num_requests_running": 0, "num_requests_waiting": 0,
                               "gpu_hit_pct": 90, "tiers": {"cpu": 0.0, "ssd": 0.0}},
            }
        )
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "overloaded"))
        assert ranking[-1] == "overloaded"
        assert set(ranking) == {"rep_a", "rep_b", "overloaded"}

    def test_empty_pool_raises(self):
        """
        Feature: route() raises RuntimeError when the replica list is empty
        Description: call route() with an empty replicas list
        Expectation: raises RuntimeError
        """
        with pytest.raises(RuntimeError):
            route([(ConstantStrategy([]), 1.0)], PROMPT_IDS, FakeRouteDataProvider({}), [])

    def test_length_mismatch_falls_back_to_random(self):
        """
        Feature: route() falls back to random order when score() returns wrong-length list
        Description: strategy returns 1 score for 2 replicas
        Expectation: returns both replica ids in some order, no exception raised
        """
        ranking = route(
            [(BadLengthStrategy(), 1.0)],
            PROMPT_IDS, FakeRouteDataProvider({}), _replicas("rep_a", "rep_b"),
        )
        assert set(ranking) == {"rep_a", "rep_b"}

    def test_strategy_exception_falls_back_to_random(self):
        """
        Feature: exceptions from score() cause route() to fall back to random order
        Description: strategy raises KeyError inside score()
        Expectation: returns all replica ids in some order, no exception raised
        """
        ranking = route([(RaisingStrategy(), 1.0)], PROMPT_IDS, FakeRouteDataProvider({}), _replicas("rep_a", "rep_b"))
        assert set(ranking) == {"rep_a", "rep_b"}

    def test_nan_score_ranked_last(self):
        """
        Feature: non-finite (NaN) scores are ranked last
        Description: first replica scores NaN; others score 0.1 and 0.5
        Expectation: NaN replica is ranked last; highest-score replica ranks first
        """
        provider = FakeRouteDataProvider({})
        ranking = route(
            [(ConstantStrategy([float("nan"), 0.1, 0.5]), 1.0)],
            PROMPT_IDS, provider, _replicas("nan_rep", "low_rep", "high_rep"),
        )
        assert ranking[0] == "high_rep"
        assert ranking[-1] == "nan_rep"


# --------------------------------------------------------------------------- #
# from_config() classmethod
# --------------------------------------------------------------------------- #
class TestFromConfig:
    def test_from_config_correct_fields(self):
        """
        Feature: from_config() transfers all config fields to the strategy instance
        Description: build a KVCAwareStrategyConfig with non-default values, then from_config()
        Expectation: strategy alpha, load_threshold, and layer_weights match the config
        """
        from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(
            alpha=0.6, load_threshold=0.15,
            layer_weights={"cpu": 0.8, "ssd": 0.1},
            weight=0.9, collector_names=["vllm_zmq"],
        )
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.alpha == pytest.approx(0.6)
        assert strat.load_threshold == pytest.approx(0.15)
        assert strat.layer_weights == {"cpu": 0.8, "ssd": 0.1}

    def test_from_config_scores_match_direct(self):
        """
        Feature: a strategy built via from_config() produces the same scores as one built directly
        Description: construct two identical strategies (one via config, one directly) and compare
        Expectation: both strategies return approx-equal score lists for the same inputs
        """
        from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(
            alpha=0.7, load_threshold=0.1,
            layer_weights={"cpu": 1.0, "ssd": 0.25},
            weight=1.0, collector_names=["vllm_zmq"],
        )
        strat_from_cfg = KVCacheAwareStrategy.from_config(cfg)
        strat_direct = _strat(alpha=0.7, load_threshold=0.1, layer_weights={"cpu": 1.0, "ssd": 0.25})
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"kv_cache_usage_perc": 0.3, "num_requests_running": 1, "num_requests_waiting": 0,
                          "gpu_hit_pct": 80, "tiers": {"cpu": 0.0, "ssd": 0.0}},
                "rep_b": {"kv_cache_usage_perc": 0.92, "num_requests_running": 0, "num_requests_waiting": 0,
                          "gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.0}},
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat_from_cfg.score(PROMPT_IDS, provider, replicas) == pytest.approx(
            strat_direct.score(PROMPT_IDS, provider, replicas)
        )
