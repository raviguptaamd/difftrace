"""
DiffTrace Engine — Main user-facing API.

Provides a high-level interface for:
1. Instrumenting dLLM inference with provenance capture
2. Querying and managing provenance records
3. Replaying inference from stored provenance
4. Running in distributed mode across multiple GPUs
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch

from difftrace.utils.config import DiffTraceConfig
from difftrace.core.trajectory import TrajectoryCapture, TrajectoryRecord
from difftrace.core.compressor import DifferentialCompressor, CompressedTrajectory
from difftrace.core.store import ProvenanceStore, ProvenanceRecord
from difftrace.core.replay import ReplayEngine, ReplayResult
from difftrace.core.async_io import AsyncProvenanceWriter
from difftrace.core.distributed import DistributedCoordinator

logger = logging.getLogger(__name__)


class DiffTraceEngine:
    """
    Main DiffTrace engine.

    Usage:
        config = DiffTraceConfig.minimal()
        engine = DiffTraceEngine(config)

        # During inference (model-specific hook handles this):
        engine.store.store_trajectory(trajectory)

        # Query provenance:
        records = engine.list_records()

        # Replay:
        result = engine.replay(request_id)
    """

    def __init__(self, config: DiffTraceConfig):
        self.config = config

        # Initialize components
        self._compressor = DifferentialCompressor(config.compression)
        self._replay_engine = ReplayEngine(self._compressor)
        self._store: Optional[ProvenanceStore] = None
        self._distributed: Optional[DistributedCoordinator] = None

        if config.enabled:
            if config.distributed.enabled:
                self._distributed = DistributedCoordinator(config)
                self._store = self._distributed.get_local_store()
            else:
                self._store = ProvenanceStore(config)

        logger.info(
            f"DiffTrace engine initialized: "
            f"capture={config.capture.granularity.value}, "
            f"compression={config.compression.mode.value}, "
            f"io={config.io.mode.value}, "
            f"distributed={config.distributed.enabled}"
        )

    @property
    def store(self) -> Optional[ProvenanceStore]:
        return self._store

    @property
    def compressor(self) -> DifferentialCompressor:
        return self._compressor

    @property
    def replay_engine(self) -> ReplayEngine:
        return self._replay_engine

    def store_trajectory(self, trajectory: TrajectoryRecord) -> Optional[ProvenanceRecord]:
        """Store a trajectory and return its index record."""
        if self._store is None:
            logger.warning("DiffTrace is disabled; trajectory not stored")
            return None

        if self._distributed is not None:
            self._distributed.store_trajectory(trajectory)
            return self._store.get_record(trajectory.request_id)
        else:
            return self._store.store_trajectory(trajectory)

    def replay(self, request_id: str) -> Optional[ReplayResult]:
        """Replay inference from stored provenance (token replay)."""
        if self._store is None:
            return None

        compressed = self._store.load_trajectory(request_id)
        if compressed is None:
            logger.error(f"Cannot replay: no trajectory found for {request_id}")
            return None

        return self._replay_engine.token_replay(compressed)

    def list_records(self, **kwargs) -> List[ProvenanceRecord]:
        """List provenance records."""
        if self._store is None:
            return []
        return self._store.list_records(**kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        stats = {"enabled": self.config.enabled}
        if self._store:
            stats["store"] = self._store.get_stats()
        if self._distributed:
            stats["distributed"] = self._distributed.get_stats()
        return stats

    def flush(self):
        """Flush all pending I/O."""
        if self._distributed:
            self._distributed.flush()
        elif self._store:
            self._store.flush()

    def shutdown(self):
        """Gracefully shut down the engine."""
        if self._distributed:
            self._distributed.shutdown()
        elif self._store:
            self._store.shutdown()
        logger.info("DiffTrace engine shut down")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
