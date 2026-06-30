"""KVCache-aware runtime strategy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from uni_agent.llm_router.collectors.metric_spec import MetricKey
from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.strategies.registry import StrategyRegistry

if TYPE_CHECKING:
    from uni_agent.llm_router.collectors.provider import RouteDataProvider
    from uni_agent.llm_router.strategies.base import ReplicaInfo

logger = get_router_logger("kvc-aware-strategy")


class StrategyError(Exception):
    """Strategy construction or scoring error."""


class KVCacheAwareStrategy:
    """Runtime strategy constructed from a ``KVCAwareStrategyConfig``."""

    def __init__(
        self,
        *,
        alpha: float,
        load_threshold: float,
        layer_weights: dict[str, float],
        collector_names: list[str],
        weight: float,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 < load_threshold < 1:
            raise StrategyError(f"load_threshold must be in (0, 1), got {load_threshold}")
        _valid_tiers = {"cpu", "ssd"}
        for tier, tier_weight in layer_weights.items():
            if tier not in _valid_tiers:
                raise StrategyError(f"layer_weights key must be in {_valid_tiers}, got '{tier}'")
            if tier_weight < 0:
                raise StrategyError(f"layer_weights[{tier}] must be >= 0, got {tier_weight}")
        self.alpha = float(alpha)
        self.load_threshold = float(load_threshold)
        self.layer_weights = dict(layer_weights)
        self.collector_names = collector_names
        self.weight = weight
        # Tiers for which we've already warned about missing slow-path data,
        # so the slow-path degradation notice is emitted once, not per request.
        self._tier_warned: set[str] = set()
        logger.info(
            f"KVCacheAwareStrategy created: alpha={self.alpha:.2f}, load_threshold={self.load_threshold:.2f}, layer_weights={self.layer_weights}",
        )

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> "KVCacheAwareStrategy":
        """Construct a strategy instance carrying its parsed config fields."""
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=cfg.layer_weights,
            collector_names=cfg.collector_names,
            weight=cfg.weight,
        )

    def score(
        self,
        prompt_ids: list[int] | None,
        provider: "RouteDataProvider",
        replicas: list["ReplicaInfo"],
    ) -> list[float]:
        """Score each replica. Larger is better.

        Combined score:
          - all overloaded:      alpha * S_cache + (1-alpha) * S_load  (slow path)
          - partial overload, overloaded replica: (1-alpha) * S_load   (cache zeroed)
          - healthy replica:     alpha * S_cache + (1-alpha) * S_load

        S_load = (1 - kv_cache_usage_perc) / (1 + running + waiting) ∈ [0, 1].
        S_cache: GPU prefix hit (fast path, 0–1) or tier-weighted hit (slow path).
        """
        if not isinstance(replicas, list):
            raise StrategyError(f"replicas must be a list, got {type(replicas).__name__}")
        if not replicas:
            return []

        effective_prompt_ids = prompt_ids or []

        # Compute load score and overload status for each replica.
        load_scores: list[float] = []
        is_overloaded: list[bool] = []
        for replica in replicas:
            replica_metrics = provider.get_metrics(replica.replica_id)
            kv_usage = replica_metrics.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0)
            running = replica_metrics.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = replica_metrics.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            s_load = (1.0 - float(kv_usage)) / (1.0 + float(running) + float(waiting))
            overloaded = s_load < self.load_threshold
            load_scores.append(s_load)
            is_overloaded.append(overloaded)
            logger.debug(
                f"score(): replica={replica.replica_id} kv={kv_usage:.3f} running={running} waiting={waiting} s_load={s_load:.4f} overloaded={overloaded}",
            )
        all_overloaded = all(is_overloaded)
        overloaded_count = sum(is_overloaded)
        if overloaded_count > 0 and not all_overloaded:
            logger.info(
                f"score(): {overloaded_count}/{len(replicas)} replicas overloaded, routing to healthy subset",
            )

        # All overloaded → slow path.
        if all_overloaded:
            logger.info(
                f"score(): all {len(replicas)} replicas overloaded (threshold={self.load_threshold:.3f}), using slow path",
            )
            return [
                self.alpha * self._slow_cache(provider, replica, effective_prompt_ids)
                + (1 - self.alpha) * load_scores[idx]
                for idx, replica in enumerate(replicas)
            ]

        force_tier_slow_path = os.getenv("UNI_AGENT_FORCE_TIER_SLOW_PATH", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        # Some healthy replicas: query GPU hit for all at once.
        # get_gpu_prefix_hit_rate returns {replica_id: 0-100}; scale to 0-1.
        gpu_hit_pct = (
            {}
            if force_tier_slow_path
            else provider.get_gpu_prefix_hit_rate(effective_prompt_ids)
        )
        gpu_hits = [
            0.0 if is_overloaded[idx]
            else gpu_hit_pct.get(replica.replica_id, 0) / 100.0
            for idx, replica in enumerate(replicas)
        ]
        use_fast = any(gpu_hits[idx] > 0 for idx in range(len(replicas)) if not is_overloaded[idx])
        if force_tier_slow_path:
            logger.debug("score(): path=slow (forced tier cache)")
        else:
            logger.debug(f"score(): path={'fast (GPU hit)' if use_fast else 'slow (tier cache)'}")

        result = []
        for idx, replica in enumerate(replicas):
            if is_overloaded[idx]:
                # Cache zeroed: overloaded replicas score < (1-alpha) * load_threshold
                result.append((1 - self.alpha) * load_scores[idx])
            else:
                s_cache = gpu_hits[idx] if use_fast else self._slow_cache(provider, replica, effective_prompt_ids)
                result.append(self.alpha * s_cache + (1 - self.alpha) * load_scores[idx])
        logger.info(
            f"score(): final scores "
            + ", ".join(f"{r.replica_id}={result[i]:.4f}" for i, r in enumerate(replicas)),
        )
        return result

    def _slow_cache(
        self,
        provider: "RouteDataProvider",
        replica: "ReplicaInfo",
        prompt_ids: list[int],
    ) -> float:
        """Weighted tier hit rate from GPU eviction table (slow path).

        cpu and ssd tiers are mutually exclusive, so the weighted sum is <= 1
        with default weights {"cpu": 1.0, "ssd": 0.25}.

        If a tier's hit rate is unavailable (``None`` — mooncake tier collector
        not yet implemented), its contribution degrades to 0 and a one-time
        WARNING is emitted per tier.
        """
        total = 0.0
        for tier, tier_weight in self.layer_weights.items():
            hit = provider.get_tier_prefix_hit_rate(replica.replica_id, prompt_ids, tier)
            if hit is None:
                if tier not in self._tier_warned:
                    logger.warning(
                        f"slow path: tier='{tier}' prefix hit unavailable "
                        f"(mooncake tier collector not implemented) — "
                        f"degrading S_cache['{tier}'] to 0",
                    )
                    self._tier_warned.add(tier)
                continue
            total += hit * tier_weight
        return total


# Auto-register: config dataclass type → runtime strategy class.
StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
