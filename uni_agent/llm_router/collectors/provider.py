"""RouteDataProvider — unified query entry point for routing decisions.

Strategy layers call ``RouteDataProvider`` methods to get metrics data.
It delegates to store instances (``MetricsStore`` for polling metrics,
``KVCacheStore`` for GPU prefix cache data) created via the registry.

Collectors and stores are created from ``BUILTIN_REGISTRY`` based on
``collection_names``.  Stores are deduplicated by class — the same
store class is only instantiated once and shared across all collectors
that reference it.

Endpoint data (server addresses, kv-event endpoints) is routed to
the appropriate collector type at creation time:
- ``PollingCollector`` → ``server_addresses``
- ``EventCollector`` subclasses → ``kv_event_addresses``
"""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.collector.polling_collector import PollingCollector
from uni_agent.llm_router.collectors.collector.event_collector import EventCollector
from uni_agent.llm_router.collectors.collector.vllm.event_collector import VLLMKVEventCollector
from uni_agent.llm_router.collectors.hash import get_prefix_hashes
from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore
from uni_agent.llm_router.collectors.store.metrics_store import MetricsStore
from uni_agent.llm_router.collectors.store.mooncake_tier_store import MooncakeTierStore
from uni_agent.llm_router.collectors.registry import BUILTIN_REGISTRY
from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("provider")


class RouteDataProvider:
    """Unified query entry point — strategies use this to access all metrics.

    ``RouteDataProvider`` creates collectors and stores. Stores are
    deduplicated by class so that e.g. all polling collectors share
    the same ``MetricsStore`` instance.

    Query computations (prefix hit rate) are implemented here — they read
    from ``KVCacheStore`` data fields.

    Args:
        collectors_config: ``CollectorConfig`` — provides common settings
            and endpoint addresses.
        collection_names: List of collection names to initialize (e.g.
            ``["vllm_metrics", "vllm_zmq"]``).
    """

    def __init__(
        self,
        collectors_config,
        collection_names,
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
        mooncake_config=None,
    ) -> None:
        self._collection_names = collection_names
        self._mooncake_config = mooncake_config
        self._mooncake_store: Any = None

        # ── Create stores (deduplicated by class) and collectors ────────
        self._stores: dict[type, Any] = {}
        self._collectors: list = []
        for name in collection_names:
            store_cls = BUILTIN_REGISTRY.get_store(name)
            if store_cls not in self._stores:
                self._stores[store_cls] = store_cls()
            collector_cls = BUILTIN_REGISTRY.get_collector(name)
            # Route endpoint data by collector type:
            # - PollingCollector → server_addresses (Prometheus polling endpoints)
            # - EventCollector subclasses → kv_event_addresses (ZMQ event endpoints)
            if issubclass(collector_cls, PollingCollector):
                self._collectors.append(
                    collector_cls(config=collectors_config, server_addresses=server_addresses)
                )
            elif issubclass(collector_cls, EventCollector):
                self._collectors.append(
                    collector_cls(config=collectors_config, kv_event_addresses=kv_event_endpoints)
                )
            else:
                self._collectors.append(collector_cls(config=collectors_config))

        # ── Wire MooncakeTierStore into VLLMKVEventCollector if both present ──
        if MooncakeTierStore in self._stores:
            tier_store = self._stores[MooncakeTierStore]
            for collector in self._collectors:
                if isinstance(collector, VLLMKVEventCollector):
                    collector.tier_store = tier_store
                    logger.info("MooncakeTierStore injected into VLLMKVEventCollector")
                    break

        # ── Connect to MooncakeDistributedStore if config provided ────────
        if mooncake_config is not None and MooncakeTierStore in self._stores:
            self._setup_mooncake_store(mooncake_config)

        logger.info(
            f"RouteDataProvider created: collection_names={collection_names}, "
            f"collectors=[{', '.join(type(c).__name__ for c in self._collectors) or '<none>'}], "
            f"stores=[{', '.join(s.__name__ for s in self._stores) or '<none>'}]",
        )

    def _setup_mooncake_store(self, mooncake_config: Any) -> None:
        """Connect MooncakeDistributedStore for batch_get_replica_desc queries."""
        try:
            from mooncake.store import MooncakeDistributedStore
            store = MooncakeDistributedStore()
            ret = store.setup(
                mooncake_config.local_hostname,
                mooncake_config.metadata_server,
                mooncake_config.global_segment_size,
                mooncake_config.local_buffer_size,
                mooncake_config.protocol,
                mooncake_config.device_name,
                mooncake_config.master_server_address,
            )
            if ret != 0:
                logger.warning(f"MooncakeDistributedStore setup returned {ret}, tier queries disabled")
                return
            self._mooncake_store = store
            logger.info("MooncakeDistributedStore connected for tier queries")
        except Exception as e:
            logger.warning(f"Failed to connect MooncakeDistributedStore: {e}, tier queries disabled")

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors, binding each to its deduplicated store."""
        logger.info(f"RouteDataProvider starting {len(self._collectors)} collector(s)")
        for name, collector in zip(self._collection_names, self._collectors):
            store_cls = BUILTIN_REGISTRY.get_store(name)
            collector.start(store=self._stores[store_cls])
            logger.info(
                f"collector started: name={name} type={type(collector).__name__} store={store_cls.__name__}",
            )

    # ── Convenience accessors for commonly-used stores ────────────────

    @property
    def _metrics_store(self) -> MetricsStore:
        """Get the shared MetricsStore instance."""
        return self._stores[MetricsStore]

    @property
    def _kv_store(self) -> KVCacheStore:
        """Get the shared KVCacheStore instance."""
        return self._stores[KVCacheStore]

    # ── Generic polling metric queries (canonical key) ──────────────────

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a polling metric by canonical key.

        Delegates to ``MetricsStore.get(node_id, key)``.

        Args:
            node_id: Target node.
            key: ``MetricKey`` constant, e.g. ``MetricKey.KV_CACHE_USAGE_PERC``.

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if
            node or metric is absent.
        """
        return self._metrics_store.get(node_id, key)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full polling metrics snapshot.

        Args:
            node_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node
            is absent in the store.
        """
        return self._metrics_store.get(node_id)

    # ── KVCache prefix hit rate queries ────────────────────────────────

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Match prefix hashes against cached blocks, return per-replica hit percent.

        Algorithm (matching aibrix ``MatchPrefix``):
            1. Compute prefix hashes via ``get_prefix_hashes``
            2. For each prefix hash, check ``replicas_by_block`` to find replicas
            3. Stop at first hash where no replica matches (chain break)
            4. Compute percent = (matched_count * 100) // total_hashes

        Args:
            prompt_ids: Current request's prompt token IDs.

        Returns:
            Dict of replica_id → prefix_match_percent (0–100).
            Empty dict if block_size is unknown or no full blocks.
        """
        kv_store = self._kv_store

        if kv_store.block_size is None:
            return {}

        prefix_hashes = get_prefix_hashes(prompt_ids, kv_store.block_size)
        if not prefix_hashes:
            return {}

        hash_strs = [str(h) for h in prefix_hashes]

        # Sequential prefix matching (aibrix MatchPrefix pattern)
        prefix_match_replicas: dict[str, int] = {}

        for i, hs in enumerate(hash_strs):
            cached_replicas = kv_store.replicas_by_block.get(hs)
            if cached_replicas is None or len(cached_replicas) == 0:
                break  # chain break — no replica caches this hash

            prefix_match_percent = (i + 1) * 100 // len(hash_strs)

            # Record percent for all replicas that have this hash
            for replica_id in cached_replicas:
                prefix_match_replicas[replica_id] = prefix_match_percent

        return prefix_match_replicas

    def get_tier_prefix_hit_rate(
        self, node_id: str, prompt_ids: list[int], tier: str,
    ) -> float | None:
        """Query tier-level prefix cache hit rate via mooncake batch_get_replica_desc.

        Resolves local xxhash prefix hashes → remote block_hash_hex → mooncake keys,
        then calls batch_get_replica_desc synchronously and classifies descriptors.

        Args:
            node_id: Target node (unused — queries are global to the mooncake store).
            prompt_ids: Current request's prompt token IDs.
            tier: ``"cpu"`` (MEMORY replicas) or ``"ssd"`` (LOCAL_DISK / NOF_SSD).

        Returns:
            Hit rate 0.0–1.0, or ``None`` when mooncake store not connected or
            no block mappings exist yet.
        """
        if self._mooncake_store is None or self._mooncake_config is None:
            return None
        if MooncakeTierStore not in self._stores:
            return None

        kv_store = self._kv_store
        if kv_store.block_size is None:
            return None

        prefix_hashes = get_prefix_hashes(prompt_ids, kv_store.block_size)
        if not prefix_hashes:
            return None

        tier_store: MooncakeTierStore = self._stores[MooncakeTierStore]
        hash_strs = [str(h) for h in prefix_hashes]
        remote_hexes = tier_store.get_remote_hexes(hash_strs)

        # Filter to blocks we have remote hex for
        available = [(hs, hex_) for hs, hex_ in zip(hash_strs, remote_hexes) if hex_ is not None]
        logger.debug(f"get_tier_prefix_hit_rate: tier={tier!r} total_blocks={len(hash_strs)} available={len(available)}")
        if not available:
            return None

        # Build mooncake keys: enumerate all tp_rank × pp_rank prefixes per block
        prefixes = self._mooncake_config.key_prefixes()
        keys: list[str] = []
        for _, hex_ in available:
            for prefix in prefixes:
                keys.append(f"{prefix}@{hex_}")

        try:
            desc_map: dict = self._mooncake_store.batch_get_replica_desc(keys)
        except Exception as e:
            logger.warning(f"batch_get_replica_desc failed: {e}")
            return None

        logger.debug(f"get_tier_prefix_hit_rate: keys={len(keys)} desc_map_hits={sum(1 for v in desc_map.values() if v)}")

        # Count blocks matching the requested tier
        hit = 0
        total = len(available)
        for _, hex_ in available:
            for prefix in prefixes:
                key = f"{prefix}@{hex_}"
                descs = desc_map.get(key)
                if not descs:
                    continue
                desc = descs[0] if isinstance(descs, (list, tuple)) else descs
                matched = _descriptor_matches_tier(desc, tier)
                if matched:
                    hit += 1
                    break  # found a match for this block, move on

        return hit / total

    def stop(self) -> None:
        """Stop all collectors and clean up."""
        logger.info(f"RouteDataProvider stopping {len(self._collectors)} collector(s)")
        for collector in self._collectors:
            collector.stop()


def _descriptor_matches_tier(desc: Any, tier: str) -> bool:
    """Return True if descriptor matches the requested tier.

    Tier mapping:
        "cpu"  → is_memory_replica()     (DRAM)
        "ssd"  → is_local_disk_replica() (local SSD) OR is_nof_replica() (NVMe-oF)
    """
    if tier == "cpu":
        m = getattr(desc, "is_memory_replica", None)
        return bool(m()) if callable(m) else False
    if tier == "ssd":
        for method in ("is_local_disk_replica", "is_nof_replica"):
            m = getattr(desc, method, None)
            if callable(m) and m():
                return True
        return False
    return False