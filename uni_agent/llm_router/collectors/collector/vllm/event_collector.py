"""
VLLMKVEventCollector — vLLM KV-cache event collector.
"""

from __future__ import annotations

import msgpack

from typing import TYPE_CHECKING

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.hash import compute_hash
from uni_agent.llm_router.collectors.collector.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.collectors.collector.zmq_event_collector import ZMQEventCollector
from uni_agent.llm_router.logging import get_router_logger

if TYPE_CHECKING:
    from uni_agent.llm_router.collectors.store.mooncake_tier_store import MooncakeTierStore

logger = get_router_logger("vllm-kv-event-collector")


class VLLMKVEventCollector(ZMQEventCollector):
    """vLLM KV-cache event collector — parses msgpack payloads,
    applies events to ``KVCacheStore``.

    ``KVCacheStore`` is passed in ``start(store)`` — owned by
    ``RouteDataProvider``.  The collector writes to it via CRUD
    methods; it never queries the store.

    Attributes:
        remote_to_local_block_hash: Mapping from vLLM remote block_hash
            to locally-computed prefix hash (str).  This is used in
            ``_apply_event`` for chained hash computation:
            - **Meaning**: vLLM event block_hash → local ``compute_hash``
              prefix hash (str form).
            - **Updated when**: BlockStored events write entries;
              BlockRemoved events delete entries.
            - **Called where**: Only in ``_apply_event``, with two paths:
              1. No ``parent_block_hash`` → ``local_parent_hash = seed``
              2. Has ``parent_block_hash`` → lookup in this mapping to
                 get the parent's local prefix hash.
    """

    def __init__(self, config, kv_event_addresses: dict[str, list[str]] | None = None) -> None:
        super().__init__(config, kv_event_addresses=kv_event_addresses)
        self.remote_to_local_block_hash: dict[str, str] = {}
        self.tier_store: MooncakeTierStore | None = None

    def _consume_payload(self, payload: bytes, node_id: str) -> None:
        """Decode msgpack payload, apply events to store.

        Args:
            payload: Raw ZMQ message bytes.
            node_id: The replica that sent this payload — used as
                     ``default_replica_id`` for ``KVCacheEvent.from_raw``.
        """
        try:
            raw_data = msgpack.unpackb(payload, raw=False)
            events = KVCacheEvent.from_raw(raw_data, default_replica_id=node_id)
            logger.debug(f"consumed payload from {node_id}: {len(events)} events")
            for event in events:
                self._apply_event(event, default_replica_id=node_id)
        except (msgpack.UnpackException, ValueError, TypeError) as e:
            logger.warning(f"failed to parse msgpack payload from {node_id}: {type(e).__name__}: {e}")

    def _apply_event(
        self,
        event: KVCacheEvent,
        default_replica_id: str | None = None,
    ) -> None:
        """Apply a KVCacheEvent to the store — vLLM backend implementation.

        For ``BlockStored`` events:
          1. Learn ``block_size`` from first event
          2. Compute local prefix hash for each block using ``compute_hash``
          3. Store local hashes in ``KVCacheStore.replicas_by_block``
          4. Map remote block_hash → local prefix hash in
             ``remote_to_local_block_hash``

        For ``BlockRemoved`` events:
          1. Convert remote block_hashes to local hashes via mapping
          2. Remove local hashes from ``KVCacheStore``
          3. Clean ``remote_to_local_block_hash`` entries

        For ``AllBlocksCleared`` events:
          1. Clear replica in ``KVCacheStore``
        """
        store = self._store
        replica_id = event.replica_id or default_replica_id or ""
        seed = 0

        if event.is_store:
            logger.debug(f"BlockStored: replica={replica_id}, blocks={len(event.block_hashes)}, block_size={event.block_size}, has_token_ids={event.token_ids is not None}, has_hex={event.block_hashes_hex is not None}")
            if store.block_size is None and event.block_size is not None:
                store.block_size = event.block_size

            if event.token_ids is not None:
                # Determine local_parent_hash for chained computation
                local_parent_hash = seed
                if event.parent_block_hash is not None:
                    local_parent_str = self.remote_to_local_block_hash.get(
                        event.parent_block_hash
                    )
                    if local_parent_str is not None:
                        local_parent_hash = int(local_parent_str)

                local_hashes: list[str] = []
                for i, block_bytes in enumerate(event.token_ids):
                    if i >= len(event.block_hashes):
                        break
                    local_hash_int = compute_hash(
                        local_parent_hash, block_bytes, seed=seed,
                    )
                    local_hash_str = str(local_hash_int)
                    bh = event.block_hashes[i]
                    self.remote_to_local_block_hash[bh] = local_hash_str
                    local_hashes.append(local_hash_str)
                    local_parent_hash = local_hash_int  # chain

                    # Write local_hash_str → block_hash_hex for mooncake tier queries.
                    if (self.tier_store is not None
                            and event.block_hashes_hex is not None
                            and i < len(event.block_hashes_hex)):
                        self.tier_store.add_mapping(local_hash_str, event.block_hashes_hex[i])

                # Store local hashes (not remote block_hashes)
                store.add_blocks(replica_id, local_hashes)

        elif event.is_remove:
            logger.debug(f"BlockRemoved: replica={replica_id}, blocks={len(event.block_hashes)}")
            # Convert remote block_hashes to local hashes for removal
            local_hashes = [
                self.remote_to_local_block_hash[bh]
                for bh in event.block_hashes
                if bh in self.remote_to_local_block_hash
            ]
            store.remove_blocks(replica_id, local_hashes)
            # Clean remote_to_local_block_hash entries
            for bh in event.block_hashes:
                self.remote_to_local_block_hash.pop(bh, None)
            if self.tier_store is not None and local_hashes:
                self.tier_store.remove_mappings(local_hashes)

        elif event.is_clear:
            logger.info(f"AllBlocksCleared: replica={replica_id}")
            store.clear_replica(replica_id)
            if self.tier_store is not None:
                # Remove all local hashes owned by this replica from tier_store.
                # KVCacheStore tracks replica→blocks, but we track the inverse here;
                # simplest is to clear all mappings whose local_hash was added by
                # this replica — use the values in remote_to_local_block_hash as proxy.
                stale = list(self.remote_to_local_block_hash.values())
                self.tier_store.remove_mappings(stale)
            self.remote_to_local_block_hash.clear()
