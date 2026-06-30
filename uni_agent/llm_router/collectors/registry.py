"""Registry — unified registry for collector classes and store classes.

Maps ``collection_name`` keys (e.g. ``"vllm_metrics"``, ``"vllm_zmq"``)
to concrete collector classes and store classes.  External callers can
register new backends via ``register_collector`` / ``register_store``.

``BUILTIN_REGISTRY`` is the module-level instance with pre-registered
built-in entries.
"""

from __future__ import annotations

from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore
from uni_agent.llm_router.collectors.store.metrics_store import MetricsStore
from uni_agent.llm_router.collectors.store.mooncake_tier_store import MooncakeTierStore
from uni_agent.llm_router.collectors.collector.vllm.event_collector import VLLMKVEventCollector
from uni_agent.llm_router.collectors.collector.vllm.polling_collector import VLLMPollingCollector
from uni_agent.llm_router.collectors.collector.mooncake_tier_collector import MooncakeTierCollector


class Registry:
    """Unified registry — collector and store classes share the same key space.

    The key is the ``collection_name`` (e.g. ``"vllm_metrics"``), which
    maps to both a collector class and a store class.  This allows
    ``RouteDataProvider`` to look up both from a single ``collection_names``
    list.
    """

    def __init__(self) -> None:
        self._collectors: dict[str, type] = {}
        self._stores: dict[str, type] = {}

    def register_collector(self, name: str, collector_cls: type) -> None:
        """Register a collector class for the given collection name."""
        self._collectors[name] = collector_cls

    def register_store(self, name: str, store_cls: type) -> None:
        """Register a store class for the given collection name."""
        self._stores[name] = store_cls

    def get_collector(self, name: str) -> type:
        """Look up a collector class by collection name.

        Raises:
            ValueError: If the name is not registered.
        """
        if name not in self._collectors:
            raise ValueError(f"Unknown collector: {name}")
        return self._collectors[name]

    def get_store(self, name: str) -> type:
        """Look up a store class by collection name.

        Raises:
            ValueError: If the name is not registered.
        """
        if name not in self._stores:
            raise ValueError(f"Unknown store: {name}")
        return self._stores[name]


# ── Module-level built-in registration ────────────────────────────────
BUILTIN_REGISTRY = Registry()
BUILTIN_REGISTRY.register_collector("vllm_metrics", VLLMPollingCollector)
BUILTIN_REGISTRY.register_collector("vllm_zmq", VLLMKVEventCollector)
BUILTIN_REGISTRY.register_collector("mooncake_tier", MooncakeTierCollector)
BUILTIN_REGISTRY.register_store("vllm_metrics", MetricsStore)
BUILTIN_REGISTRY.register_store("vllm_zmq", KVCacheStore)
BUILTIN_REGISTRY.register_store("mooncake_tier", MooncakeTierStore)
