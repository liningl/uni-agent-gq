"""KVCAwareBalancer — top-level orchestration shell for the KVCAware router.

A **pure framework shell** (detailed_balancer.md §1): it wires Config /
Strategy / collectors, manages their lifecycle, and delegates each request to
``route()``. It contains no routing algorithm.

VeRL imports this class by FQN (``router_class``) and wraps it with
``ray.remote(...)`` at runtime, so this is a plain class — directly
constructible and unit-testable. It satisfies the ``RequestLoadBalancer``
Protocol (6 methods) via structural subtyping.
"""

from __future__ import annotations

from typing import Any

# Import RouteDataProvider from its definition module (collectors.provider),
# not the ``collectors`` package attribute. Unit tests in test_balancer.py
# monkeypatch ``collectors.RouteDataProvider`` to a fake; if we imported via the
# package, Ray worker processes that fork from the patched pytest process would
# bind the fake and report "_FakeProvider" in get_status(). Sourcing the class
# from its definition site is immune to that package-attribute patch.
from uni_agent.llm_router.collectors.provider import RouteDataProvider
from uni_agent.llm_router.config import KVCAwareConfig
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.strategies import ReplicaInfo, StrategyRegistry, route

logger = get_router_logger("balancer")


class KVCAwareBalancer:
    """Pure-framework router shell. See module docstring."""

    def __init__(self, servers: dict[str, Any], router_config: Any) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        self._config = KVCAwareConfig.from_config(router_config)
        self._strategies: list[tuple[Any, float]] = [
            (StrategyRegistry.get(type(cfg)).from_config(cfg), cfg.weight)
            for cfg in self._config.strategies
        ]
        self._servers: dict[str, Any] = dict(servers)
        self._route_calls = 0
        self._init_provider()

    def _init_provider(self) -> None:
        """Resolve per-server endpoints from Ray actor handles and init the provider.

        Iterates ``self._servers``, calling ``get_server_address.remote()`` and
        ``get_kv_events_endpoints.remote()`` on each handle to dynamically
        discover the Prometheus polling addresses and ZMQ kv-event endpoints.
        The resolved addresses are then passed to ``RouteDataProvider``, which
        routes them to the appropriate collector type at creation time.

        Handles that are not real Ray actors (e.g. plain strings passed by
        unit tests or bring-up stubs) have no ``get_server_address`` remote;
        for those, dynamic discovery is skipped and collectors fall back to
        their configured/default endpoints.
        """
        import ray

        collection_names = sorted(
            {name for cfg in self._config.strategies for name in cfg.collector_names}
        )
        server_addresses: dict[str, str] = {}
        kv_event_endpoints: dict[str, list[str]] = {}
        for replica_id, handle in self._servers.items():
            if not hasattr(handle, "get_server_address"):
                logger.warning(
                    f"server '{replica_id}' handle has no get_server_address remote "
                    f"(type={type(handle).__name__}); skipping dynamic endpoint discovery",
                )
                continue
            ip, port = ray.get(handle.get_server_address.remote())
            server_addresses[replica_id] = f"{ip}:{port}"
            endpoints = ray.get(handle.get_kv_events_endpoints.remote())
            if endpoints is not None:
                kv_event_endpoints[replica_id] = endpoints
        self._provider = RouteDataProvider(
            self._config.collector, collection_names,
            server_addresses=server_addresses,
            kv_event_endpoints=kv_event_endpoints,
            mooncake_config=self._config.mooncake,
        )
        self._provider.start()

    def get_all_servers(self) -> list[str]:
        """List all active server ids."""
        return list(self._servers.keys())

    def get_status(self) -> dict:
        """Return construction + routing state for debugging.

        Reports what the balancer was wired with (pool, provider type,
        materialized strategies) and how many routing decisions it has made —
        enough to verify the construction flow over the remote boundary.
        """
        return {
            "servers": list(self._servers.keys()),
            "provider": type(self._provider).__name__,
            "strategies": [{"type": type(s).__name__, "weight": w} for s, w in self._strategies],
            "route_calls": self._route_calls,
        }

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes. No-op in v1 (no inflight)."""

    def acquire_server(
        self, request_id: str, prompt_ids: list[int] | None = None
    ) -> tuple[str, Any]:
        """Acquire the best server for a request: delegate to ``route()``, map back.

        Builds ``ReplicaInfo`` candidates from the pool, asks ``route()`` for a
        best-first ranking, and returns ``(ranking[0], handle)``. Raises
        ``RuntimeError`` if no replica is available (empty pool or all blacklisted).
        """
        replicas = [ReplicaInfo(replica_id=sid) for sid in self._servers]
        self._route_calls += 1
        ranking = route(self._strategies, prompt_ids, self._provider, replicas)
        if not ranking:
            raise RuntimeError("no available replica to route to")
        server_id = ranking[0]
        logger.info(
            f"request={request_id} routed to server={server_id} (ranking={ranking}, pool={list(self._servers)})",
        )
        return server_id, self._servers[server_id]

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the pool.

        Note: the provider is keyed by the endpoint addresses resolved at
        init time, not by this pool, so it is not touched here.
        """
        for sid, handle in servers.items():
            self._servers[sid] = handle

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers from the pool (provider is not keyed by the pool)."""
        for sid in server_ids:
            self._servers.pop(sid, None)
