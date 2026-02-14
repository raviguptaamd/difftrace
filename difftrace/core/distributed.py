"""
Distributed Provenance Coordination Module.

Manages provenance capture across multiple GPUs/nodes in distributed
inference settings (e.g., tensor-parallel or pipeline-parallel dLLM inference).

Design:
- Each rank captures its local provenance independently
- The coordinator rank collects metadata for indexing
- Provenance files are stored locally on each node
- Replay requires the same distributed configuration
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.distributed as dist

from difftrace.core.trajectory import TrajectoryRecord, TrajectoryCapture
from difftrace.core.compressor import CompressedTrajectory, DifferentialCompressor
from difftrace.core.store import ProvenanceStore
from difftrace.utils.config import DiffTraceConfig, DistributedConfig

logger = logging.getLogger(__name__)


@dataclass
class DistributedProvenanceMetadata:
    """Metadata for a distributed provenance record spanning multiple ranks."""
    request_id: str
    world_size: int
    rank_records: Dict[int, str]  # rank -> filepath mapping
    timestamp_ns: int = 0
    model_name: str = ""
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0


class DistributedCoordinator:
    """
    Coordinates provenance capture across distributed inference ranks.

    In tensor-parallel dLLM inference, each GPU processes a shard of the
    model. This coordinator ensures:
    1. Consistent request_ids across ranks
    2. Synchronized trajectory boundaries
    3. Aggregated provenance metadata
    4. Coordinated replay
    """

    def __init__(self, config: DiffTraceConfig):
        self.config = config
        self.dist_config = config.distributed

        if not self.dist_config.enabled:
            raise ValueError(
                "DistributedCoordinator requires distributed config to be enabled"
            )

        self._rank = self.dist_config.rank
        self._world_size = self.dist_config.world_size
        self._is_coordinator = (self._rank == self.dist_config.coordinator_rank)

        # Each rank has its own local store
        rank_store_path = Path(config.store.base_path) / f"rank_{self._rank}"
        rank_config = DiffTraceConfig(
            capture=config.capture,
            compression=config.compression,
            io=config.io,
            store=config.store.__class__(base_path=rank_store_path),
            distributed=config.distributed,
            enabled=config.enabled,
        )
        self._local_store = ProvenanceStore(rank_config)

        # Coordinator maintains global index
        self._global_index: Dict[str, DistributedProvenanceMetadata] = {}

    def store_trajectory(self, trajectory: TrajectoryRecord) -> None:
        """
        Store a trajectory from the local rank.

        After local storage, the coordinator rank collects metadata
        from all ranks to build the global index.
        """
        # Store locally
        record = self._local_store.store_trajectory(trajectory)

        # Synchronize metadata across ranks
        if dist.is_initialized():
            self._sync_metadata(
                trajectory.request_id,
                record.filepath,
                record.original_size_bytes,
                record.compressed_size_bytes,
            )

    def _sync_metadata(
        self,
        request_id: str,
        filepath: str,
        original_bytes: int,
        compressed_bytes: int,
    ):
        """Gather metadata from all ranks at the coordinator."""
        # Prepare local metadata
        local_meta = {
            "rank": self._rank,
            "filepath": filepath,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
        }
        meta_str = json.dumps(local_meta)

        # Gather all metadata at coordinator
        if dist.is_initialized():
            gathered = [None] * self._world_size
            dist.all_gather_object(gathered, meta_str)

            if self._is_coordinator:
                rank_records = {}
                total_orig = 0
                total_comp = 0
                for meta_json in gathered:
                    meta = json.loads(meta_json)
                    rank_records[meta["rank"]] = meta["filepath"]
                    total_orig += meta["original_bytes"]
                    total_comp += meta["compressed_bytes"]

                self._global_index[request_id] = DistributedProvenanceMetadata(
                    request_id=request_id,
                    world_size=self._world_size,
                    rank_records=rank_records,
                    timestamp_ns=time.time_ns(),
                    total_original_bytes=total_orig,
                    total_compressed_bytes=total_comp,
                )

    def get_distributed_record(
        self,
        request_id: str,
    ) -> Optional[DistributedProvenanceMetadata]:
        """Get distributed provenance metadata (coordinator only)."""
        if not self._is_coordinator:
            logger.warning(
                "get_distributed_record should only be called on coordinator"
            )
        return self._global_index.get(request_id)

    def get_local_store(self) -> ProvenanceStore:
        """Get the local provenance store for this rank."""
        return self._local_store

    def get_stats(self) -> Dict[str, Any]:
        """Get distributed provenance statistics."""
        local_stats = self._local_store.get_stats()
        stats = {
            "rank": self._rank,
            "world_size": self._world_size,
            "is_coordinator": self._is_coordinator,
            "local_stats": local_stats,
        }
        if self._is_coordinator:
            stats["global_records"] = len(self._global_index)
            stats["global_total_original_bytes"] = sum(
                m.total_original_bytes for m in self._global_index.values()
            )
            stats["global_total_compressed_bytes"] = sum(
                m.total_compressed_bytes for m in self._global_index.values()
            )
        return stats

    def flush(self):
        """Flush local store and synchronize."""
        self._local_store.flush()
        if dist.is_initialized():
            dist.barrier()

    def shutdown(self):
        """Shut down the distributed coordinator."""
        self.flush()
        self._local_store.shutdown()
