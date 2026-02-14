"""
DiffTrace configuration module.

Defines all configuration dataclasses for the DiffTrace framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class CaptureGranularity(Enum):
    """Granularity of trajectory capture."""
    TOKENS_ONLY = "tokens_only"       # Only capture sampled tokens per step
    TOKENS_AND_MASKS = "tokens_masks"  # Tokens + mask state per step
    FULL_LOGITS = "full_logits"        # Full logit tensor per step (most expensive)


class CompressionMode(Enum):
    """Compression mode for provenance data."""
    NONE = "none"                  # No compression
    DIFF_LOSSLESS = "diff_lossless"  # Differential + lossless (zstd)
    DIFF_LOSSY = "diff_lossy"        # Differential + error-bounded lossy for logits


class IOMode(Enum):
    """I/O mode for provenance writing."""
    SYNC = "sync"       # Synchronous (blocking)
    ASYNC = "async"     # Asynchronous (non-blocking, lazy flush)


@dataclass
class CaptureConfig:
    """Configuration for trajectory capture."""
    granularity: CaptureGranularity = CaptureGranularity.TOKENS_AND_MASKS
    capture_random_state: bool = True
    capture_timestamps: bool = True
    capture_every_n_steps: int = 1  # Capture every N denoising steps


@dataclass
class CompressionConfig:
    """Configuration for differential compression."""
    mode: CompressionMode = CompressionMode.DIFF_LOSSLESS
    zstd_level: int = 3  # zstd compression level (1-22)
    lossy_error_bound: float = 1e-3  # Error bound for lossy logit compression
    lossy_error_mode: str = "abs"  # "abs" or "rel" error bound


@dataclass
class IOConfig:
    """Configuration for async I/O."""
    mode: IOMode = IOMode.ASYNC
    num_workers: int = 2
    buffer_size_mb: int = 64  # Double buffer size in MB
    flush_interval_ms: int = 100  # Lazy flush interval
    max_queue_size: int = 1024


@dataclass
class StoreConfig:
    """Configuration for provenance store."""
    base_path: Path = field(default_factory=lambda: Path("./provenance_store"))
    max_records: int = 100000
    index_in_memory: bool = True


@dataclass
class DistributedConfig:
    """Configuration for distributed provenance."""
    enabled: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"  # nccl or gloo
    coordinator_rank: int = 0


@dataclass
class DiffTraceConfig:
    """Top-level DiffTrace configuration."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    io: IOConfig = field(default_factory=IOConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    enabled: bool = True  # Master switch

    @classmethod
    def minimal(cls) -> "DiffTraceConfig":
        """Minimal overhead configuration: tokens only, async I/O, diff compression."""
        return cls(
            capture=CaptureConfig(
                granularity=CaptureGranularity.TOKENS_ONLY,
                capture_random_state=True,
            ),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS),
            io=IOConfig(mode=IOMode.ASYNC),
        )

    @classmethod
    def full(cls) -> "DiffTraceConfig":
        """Full provenance: logits, all states, async I/O, lossy compression."""
        return cls(
            capture=CaptureConfig(
                granularity=CaptureGranularity.FULL_LOGITS,
                capture_random_state=True,
                capture_timestamps=True,
            ),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSY),
            io=IOConfig(mode=IOMode.ASYNC),
        )

    @classmethod
    def debug(cls) -> "DiffTraceConfig":
        """Debug mode: full capture, no compression, sync I/O."""
        return cls(
            capture=CaptureConfig(
                granularity=CaptureGranularity.FULL_LOGITS,
                capture_random_state=True,
                capture_timestamps=True,
            ),
            compression=CompressionConfig(mode=CompressionMode.NONE),
            io=IOConfig(mode=IOMode.SYNC),
        )
