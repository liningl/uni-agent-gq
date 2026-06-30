"""
MooncakeCollectorConfig — config for mooncake tier collector.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MooncakeCollectorConfig:
    """Connection and key-construction parameters for the mooncake tier collector.

    model_name must equal the last path segment of the vllm model path
    (same as vllm worker.py: model_config.model.rstrip("/").split("/")[-1]).

    tp_size * pp_size key prefixes are enumerated per block hash.
    pcp_rank, dcp_rank, group_id are hardcoded to 0 (verl does not expose them).
    """

    model_name: str
    metadata_server: str
    master_server_address: str
    tp_size: int = 1
    pp_size: int = 1
    cache_prefix: str = ""
    local_hostname: str = "localhost"
    global_segment_size: int = 67108864    # 64 MB
    local_buffer_size: int = 134217728     # 128 MB
    protocol: str = "tcp"
    device_name: str = ""

    def key_prefixes(self) -> list[str]:
        """Return all tp_rank x pp_rank key prefix strings."""
        prefixes = []
        prefix_base = f"{self.cache_prefix}@" if self.cache_prefix else ""
        for tp in range(self.tp_size):
            for pp in range(self.pp_size):
                prefixes.append(
                    f"{prefix_base}{self.model_name}"
                    f"@tp_rank:{tp}"
                    f"@pcp0"
                    f"@dcp0"
                    f"@pp_rank:{pp}"
                    f"@group:0"
                )
        return prefixes
