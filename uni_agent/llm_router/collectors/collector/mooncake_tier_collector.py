"""
MooncakeTierCollector — no-op collector stub for registry integration.

The actual mooncake batch_get_replica_desc query happens synchronously in
RouteDataProvider.get_tier_prefix_hit_rate(), not in a background collector.
This stub satisfies the Registry's collector requirement so "mooncake_tier"
can be listed in collection_names without error.
"""

from __future__ import annotations

from typing import Any


class MooncakeTierCollector:
    """No-op collector stub — satisfies Registry interface, does no background work."""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        pass

    def start(self, store: Any) -> None:
        pass

    def stop(self) -> None:
        pass
