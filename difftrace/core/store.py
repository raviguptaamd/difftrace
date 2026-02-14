"""
Provenance Store Module.

Manages persistent storage and retrieval of provenance records.
Supports indexed lookup by request_id, model, time range, etc.
"""

import os
import json
import time
import pickle
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass, field
import logging

from difftrace.core.trajectory import TrajectoryRecord
from difftrace.core.compressor import CompressedTrajectory, DifferentialCompressor
from difftrace.core.async_io import AsyncProvenanceWriter
from difftrace.utils.config import (
    DiffTraceConfig,
    StoreConfig,
    CompressionConfig,
    IOConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """
    A queryable provenance record entry.

    This is the lightweight index entry; the full compressed trajectory
    is stored on disk and loaded on demand.
    """
    request_id: str
    model_name: str
    timestamp_ns: int
    num_steps: int
    sequence_length: int
    original_size_bytes: int
    compressed_size_bytes: int
    filepath: str
    prompt_text: str = ""
    generated_text: str = ""
    device: str = ""
    world_size: int = 1
    rank: int = 0


class ProvenanceStore:
    """
    Main provenance store with indexing and retrieval.

    Features:
    - In-memory index for fast queries
    - On-disk compressed trajectory storage
    - Integrated compression and async I/O
    - Thread-safe operations
    """

    def __init__(self, config: DiffTraceConfig):
        self.config = config
        self.store_config = config.store
        self.base_path = Path(self.store_config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._compressor = DifferentialCompressor(config.compression)
        self._writer = AsyncProvenanceWriter(
            config=config.io,
            output_dir=self.base_path / "trajectories",
            on_write_complete=self._on_write_complete,
        )

        # In-memory index
        self._index: Dict[str, ProvenanceRecord] = {}
        self._index_lock = threading.Lock()

        # Load existing index if present
        self._load_index()

    def store_trajectory(self, trajectory: TrajectoryRecord) -> ProvenanceRecord:
        """
        Compress and store a trajectory, returning its index record.

        This is the main entry point for storing provenance data.
        """
        # Compress
        compressed = self._compressor.compress_trajectory(trajectory)

        # Create index record
        filepath = str(
            self.base_path / "trajectories" / f"{trajectory.request_id}.dtrace"
        )
        record = ProvenanceRecord(
            request_id=trajectory.request_id,
            model_name=trajectory.model_name,
            timestamp_ns=trajectory.start_time_ns,
            num_steps=len(compressed.steps),
            sequence_length=trajectory.sequence_length,
            original_size_bytes=compressed.original_size_bytes,
            compressed_size_bytes=compressed.compressed_size_bytes,
            filepath=filepath,
            prompt_text=trajectory.prompt_text[:200],  # Truncate for index
            generated_text=trajectory.generated_text[:200],
            device=trajectory.device,
            world_size=trajectory.world_size,
            rank=trajectory.rank,
        )

        # Add to index
        with self._index_lock:
            self._index[trajectory.request_id] = record

        # Write async
        self._writer.write_trajectory(compressed)

        return record

    def load_trajectory(self, request_id: str) -> Optional[CompressedTrajectory]:
        """Load a compressed trajectory from disk."""
        with self._index_lock:
            record = self._index.get(request_id)

        if record is None:
            logger.warning(f"No provenance record found for {request_id}")
            return None

        filepath = Path(record.filepath)
        if not filepath.exists():
            logger.error(f"Provenance file not found: {filepath}")
            return None

        with open(filepath, "rb") as f:
            data = f.read()
        return AsyncProvenanceWriter.deserialize(data)

    def get_record(self, request_id: str) -> Optional[ProvenanceRecord]:
        """Get an index record by request_id."""
        with self._index_lock:
            return self._index.get(request_id)

    def list_records(
        self,
        model_name: Optional[str] = None,
        time_range: Optional[tuple] = None,
        limit: int = 100,
    ) -> List[ProvenanceRecord]:
        """List provenance records with optional filters."""
        with self._index_lock:
            records = list(self._index.values())

        if model_name:
            records = [r for r in records if r.model_name == model_name]

        if time_range:
            start_ns, end_ns = time_range
            records = [
                r for r in records
                if start_ns <= r.timestamp_ns <= end_ns
            ]

        # Sort by timestamp (most recent first)
        records.sort(key=lambda r: r.timestamp_ns, reverse=True)
        return records[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._index_lock:
            num_records = len(self._index)
            total_original = sum(r.original_size_bytes for r in self._index.values())
            total_compressed = sum(
                r.compressed_size_bytes for r in self._index.values()
            )

        io_stats = self._writer.stats
        return {
            "num_records": num_records,
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "overall_compression_ratio": (
                total_original / total_compressed if total_compressed > 0 else 0
            ),
            "io_stats": {
                "total_writes": io_stats.total_writes,
                "total_bytes_written": io_stats.total_bytes_written,
                "avg_write_latency_ms": io_stats.avg_write_latency_ms,
                "throughput_mb_s": io_stats.throughput_mb_s,
                "queue_drops": io_stats.queue_drops,
            },
        }

    def flush(self):
        """Flush pending writes and save index."""
        self._writer.flush_and_wait()
        self._save_index()

    def shutdown(self):
        """Gracefully shut down the store."""
        self.flush()
        self._writer.shutdown()

    def _on_write_complete(self, request_id: str, success: bool):
        """Callback when an async write completes."""
        if not success:
            logger.error(f"Failed to write provenance for {request_id}")

    def _save_index(self):
        """Save the in-memory index to disk."""
        index_path = self.base_path / "index.json"
        with self._index_lock:
            records = {
                rid: {
                    "request_id": r.request_id,
                    "model_name": r.model_name,
                    "timestamp_ns": r.timestamp_ns,
                    "num_steps": r.num_steps,
                    "sequence_length": r.sequence_length,
                    "original_size_bytes": r.original_size_bytes,
                    "compressed_size_bytes": r.compressed_size_bytes,
                    "filepath": r.filepath,
                    "prompt_text": r.prompt_text,
                    "generated_text": r.generated_text,
                    "device": r.device,
                    "world_size": r.world_size,
                    "rank": r.rank,
                }
                for rid, r in self._index.items()
            }
        with open(index_path, "w") as f:
            json.dump(records, f, indent=2)

    def _load_index(self):
        """Load index from disk if it exists."""
        index_path = self.base_path / "index.json"
        if not index_path.exists():
            return

        try:
            with open(index_path, "r") as f:
                records = json.load(f)
            with self._index_lock:
                for rid, rdata in records.items():
                    self._index[rid] = ProvenanceRecord(**rdata)
            logger.info(f"Loaded {len(self._index)} provenance records from index")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
