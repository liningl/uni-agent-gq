"""
MooncakeTierStore — thread-safe store mapping local prefix hashes to
mooncake replica tiers (cpu / ssd / disk).

Written by VLLMKVEventCollector (ZMQ thread), read by
RouteDataProvider.get_tier_prefix_hit_rate() (Ray actor thread).
"""

from __future__ import annotations

import threading


class MooncakeTierStore:
    """Maps local xxhash strings to mooncake remote block_hash_hex values,
    and caches the tier label returned by batch_get_replica_desc.

    Attributes:
        local_to_remote_hex: local_hash_str -> block_hash_hex (bytes.hex()).
            Populated when VLLMKVEventCollector processes BlockStored events
            with block_hashes_hex present (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.local_to_remote_hex: dict[str, str] = {}

    def add_mapping(self, local_hash_str: str, block_hash_hex: str) -> None:
        with self._lock:
            self.local_to_remote_hex[local_hash_str] = block_hash_hex

    def remove_mappings(self, local_hash_strs: list[str]) -> None:
        with self._lock:
            for h in local_hash_strs:
                self.local_to_remote_hex.pop(h, None)

    def clear_replica(self, replica_id: str, local_hashes: list[str]) -> None:
        self.remove_mappings(local_hashes)

    def get_remote_hex(self, local_hash_str: str) -> str | None:
        with self._lock:
            return self.local_to_remote_hex.get(local_hash_str)

    def get_remote_hexes(self, local_hash_strs: list[str]) -> list[str | None]:
        with self._lock:
            return [self.local_to_remote_hex.get(h) for h in local_hash_strs]
