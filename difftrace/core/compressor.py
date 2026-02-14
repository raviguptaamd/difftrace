"""
Differential Compressor Module.

Compresses denoising trajectories using differential encoding:
- Consecutive denoising states are highly correlated
- Store only the delta (changed positions + new values)
- Apply zstd for lossless or error-bounded lossy compression for logits

Inspired by LibPressio (Underwood et al.) for error-bounded compression
and DataStates-LLM (Nicolae et al.) for lazy differential checkpointing.
"""

import struct
import zlib
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from difftrace.core.trajectory import DenoiseStep, TrajectoryRecord
from difftrace.utils.config import CompressionConfig, CompressionMode


# ─── Wire format constants ───────────────────────────────────────────────────
MAGIC = b"DTRC"  # DiffTrace Record
VERSION = 1
HEADER_FMT = "<4sHHI"  # magic, version, flags, num_steps
STEP_HEADER_FMT = "<iiqII"  # step_index, num_unmasked, timestamp_ns, positions_bytes, tokens_bytes


@dataclass
class CompressedStep:
    """A compressed denoising step."""
    step_index: int
    num_unmasked: int
    timestamp_ns: int
    # Differential-encoded: only changed positions and their new tokens
    positions_data: bytes     # Compressed positions array
    tokens_data: bytes        # Compressed tokens array
    mask_diff_data: Optional[bytes] = None   # Compressed mask diff (XOR with previous)
    logits_data: Optional[bytes] = None      # Compressed logits (lossy or lossless)
    confidence_data: Optional[bytes] = None  # Compressed confidence scores
    rng_state_data: Optional[bytes] = None   # Serialized RNG state

    def size_bytes(self) -> int:
        total = struct.calcsize(STEP_HEADER_FMT)
        total += len(self.positions_data)
        total += len(self.tokens_data)
        if self.mask_diff_data:
            total += len(self.mask_diff_data)
        if self.logits_data:
            total += len(self.logits_data)
        if self.confidence_data:
            total += len(self.confidence_data)
        if self.rng_state_data:
            total += len(self.rng_state_data)
        return total


@dataclass
class CompressedTrajectory:
    """A fully compressed trajectory."""
    request_id: str
    metadata: Dict[str, Any]   # All non-step metadata
    steps: List[CompressedStep]

    # Compression stats
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.compressed_size_bytes == 0:
            return 0.0
        return self.original_size_bytes / self.compressed_size_bytes


class DifferentialCompressor:
    """
    Differential compression engine for denoising trajectories.

    Key insight: In masked diffusion models, consecutive denoising steps
    differ only in the newly unmasked positions. By storing only the diff,
    we achieve high compression ratios without lossy approximation for
    token-level data.

    For full logits, we apply error-bounded lossy compression inspired
    by LibPressio, using quantization + zstd to achieve configurable
    accuracy-storage tradeoffs.
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        self._zstd_compressor = None
        self._zstd_decompressor = None
        if HAS_ZSTD and config.mode != CompressionMode.NONE:
            self._zstd_compressor = zstd.ZstdCompressor(level=config.zstd_level)
            self._zstd_decompressor = zstd.ZstdDecompressor()

    def compress_trajectory(self, trajectory: TrajectoryRecord) -> CompressedTrajectory:
        """Compress a full trajectory using differential encoding."""
        original_size = trajectory.total_size_bytes()

        # Compress each step
        compressed_steps = []
        prev_mask = None

        for step in trajectory.steps:
            cstep = self._compress_step(step, prev_mask)
            compressed_steps.append(cstep)
            if step.mask_state is not None:
                prev_mask = step.mask_state

        # Build metadata dict
        metadata = {
            "request_id": trajectory.request_id,
            "prompt_tokens": trajectory.prompt_tokens.tobytes(),
            "prompt_text": trajectory.prompt_text,
            "model_name": trajectory.model_name,
            "model_config": trajectory.model_config,
            "num_diffusion_steps": trajectory.num_diffusion_steps,
            "sequence_length": trajectory.sequence_length,
            "temperature": trajectory.temperature,
            "top_p": trajectory.top_p,
            "device": trajectory.device,
            "gpu_name": trajectory.gpu_name,
            "world_size": trajectory.world_size,
            "rank": trajectory.rank,
            "start_time_ns": trajectory.start_time_ns,
            "end_time_ns": trajectory.end_time_ns,
        }
        if trajectory.generated_tokens is not None:
            metadata["generated_tokens"] = trajectory.generated_tokens.tobytes()
        metadata["generated_text"] = trajectory.generated_text

        # Serialize initial RNG state
        if trajectory.initial_rng_state is not None:
            metadata["initial_rng_state"] = self._serialize_rng_state(
                trajectory.initial_rng_state
            )

        compressed = CompressedTrajectory(
            request_id=trajectory.request_id,
            metadata=metadata,
            steps=compressed_steps,
            original_size_bytes=original_size,
        )
        compressed.compressed_size_bytes = sum(s.size_bytes() for s in compressed_steps)
        return compressed

    def _compress_step(
        self,
        step: DenoiseStep,
        prev_mask: Optional[np.ndarray] = None,
    ) -> CompressedStep:
        """Compress a single denoising step using differential encoding."""

        if self.config.mode == CompressionMode.NONE:
            return self._compress_step_raw(step)

        # --- Token positions and values (lossless differential) ---
        positions_data = self._compress_array(step.unmasked_positions)
        tokens_data = self._compress_array(step.sampled_tokens)

        # --- Mask diff (XOR with previous for minimal diff) ---
        mask_diff_data = None
        if step.mask_state is not None:
            if prev_mask is not None:
                # XOR diff: only stores changed bits
                diff = np.bitwise_xor(
                    step.mask_state.view(np.uint8),
                    prev_mask.view(np.uint8),
                )
                mask_diff_data = self._compress_array(diff)
            else:
                mask_diff_data = self._compress_array(
                    step.mask_state.view(np.uint8)
                )

        # --- Logits (lossy or lossless depending on config) ---
        logits_data = None
        if step.logits is not None:
            if self.config.mode == CompressionMode.DIFF_LOSSY:
                logits_data = self._compress_logits_lossy(step.logits)
            else:
                logits_data = self._compress_array(step.logits)

        # --- Confidence scores ---
        confidence_data = None
        if step.confidence_scores is not None:
            confidence_data = self._compress_array(step.confidence_scores)

        # --- RNG state ---
        rng_state_data = None
        if step.rng_state is not None:
            rng_state_data = self._serialize_rng_state(step.rng_state)

        return CompressedStep(
            step_index=step.step_index,
            num_unmasked=step.num_unmasked,
            timestamp_ns=step.timestamp_ns,
            positions_data=positions_data,
            tokens_data=tokens_data,
            mask_diff_data=mask_diff_data,
            logits_data=logits_data,
            confidence_data=confidence_data,
            rng_state_data=rng_state_data,
        )

    def _compress_step_raw(self, step: DenoiseStep) -> CompressedStep:
        """Store step without compression (debug mode)."""
        return CompressedStep(
            step_index=step.step_index,
            num_unmasked=step.num_unmasked,
            timestamp_ns=step.timestamp_ns,
            positions_data=(
                step.unmasked_positions.tobytes()
                if step.unmasked_positions is not None else b""
            ),
            tokens_data=(
                step.sampled_tokens.tobytes()
                if step.sampled_tokens is not None else b""
            ),
            mask_diff_data=(
                step.mask_state.tobytes()
                if step.mask_state is not None else None
            ),
            logits_data=(
                step.logits.tobytes()
                if step.logits is not None else None
            ),
            confidence_data=(
                step.confidence_scores.tobytes()
                if step.confidence_scores is not None else None
            ),
            rng_state_data=(
                self._serialize_rng_state(step.rng_state)
                if step.rng_state is not None else None
            ),
        )

    def _compress_array(self, arr: Optional[np.ndarray]) -> bytes:
        """Compress a numpy array using zstd."""
        if arr is None:
            return b""
        raw = arr.tobytes()
        if self._zstd_compressor is not None:
            return self._zstd_compressor.compress(raw)
        return zlib.compress(raw, level=6)

    def decompress_array(
        self,
        data: bytes,
        dtype: np.dtype,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        """Decompress a numpy array."""
        if len(data) == 0:
            return np.array([], dtype=dtype)
        if self._zstd_decompressor is not None:
            raw = self._zstd_decompressor.decompress(data)
        else:
            raw = zlib.decompress(data)
        arr = np.frombuffer(raw, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr

    def _compress_logits_lossy(self, logits: np.ndarray) -> bytes:
        """
        Error-bounded lossy compression for logit tensors.

        Uses linear quantization with configurable error bound,
        followed by zstd compression of the quantized representation.

        Inspired by SZ and LibPressio error-bounded compression.
        """
        error_bound = self.config.lossy_error_bound

        if self.config.lossy_error_mode == "abs":
            # Absolute error-bounded quantization
            vmin = logits.min()
            vmax = logits.max()
            value_range = vmax - vmin

            if value_range < 1e-10:
                # Near-constant tensor
                header = struct.pack("<ffi", vmin, 0.0, 0)
                return self._compress_bytes(header)

            # Quantization levels = range / (2 * error_bound)
            raw_levels = value_range / (2 * error_bound)
            if not np.isfinite(raw_levels) or raw_levels > 1e9:
                raw_levels = 65535
            num_levels = max(int(raw_levels), 1)
            num_levels = min(num_levels, 65535)  # Fit in uint16
            scale = value_range / num_levels

            # Quantize to uint16
            quantized = ((logits - vmin) / scale).clip(0, num_levels).astype(np.uint16)

            # Pack: header (vmin, scale, num_levels) + quantized data
            header = struct.pack("<ffi", float(vmin), float(scale), num_levels)
            payload = header + quantized.tobytes()

        else:
            # Relative error-bounded: use log-space quantization
            abs_logits = np.abs(logits)
            sign = np.sign(logits).astype(np.int8)
            log_abs = np.log1p(abs_logits).astype(np.float32)

            vmin = log_abs.min()
            vmax = log_abs.max()
            value_range = vmax - vmin
            num_levels = max(int(value_range / (2 * error_bound)), 1)
            num_levels = min(num_levels, 65535)
            scale = value_range / num_levels if num_levels > 0 else 1.0

            quantized = ((log_abs - vmin) / scale).clip(0, num_levels).astype(np.uint16)
            header = struct.pack("<ffii", float(vmin), float(scale), num_levels, 1)
            payload = header + sign.tobytes() + quantized.tobytes()

        return self._compress_bytes(payload)

    def decompress_logits_lossy(
        self,
        data: bytes,
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Decompress lossy-compressed logits."""
        payload = self._decompress_bytes(data)

        # Check if relative mode (header has 4th int = 1)
        header_size = struct.calcsize("<ffi")
        if len(payload) > header_size:
            # Try reading extended header
            try:
                vmin, scale, num_levels, mode = struct.unpack_from("<ffii", payload)
                if mode == 1:
                    # Relative mode
                    offset = struct.calcsize("<ffii")
                    num_elements = 1
                    for s in shape:
                        num_elements *= s
                    sign = np.frombuffer(
                        payload[offset:offset + num_elements], dtype=np.int8
                    )
                    quantized = np.frombuffer(
                        payload[offset + num_elements:], dtype=np.uint16
                    ).reshape(shape)
                    log_abs = quantized.astype(np.float32) * scale + vmin
                    abs_vals = np.expm1(log_abs)
                    return (sign.reshape(shape).astype(np.float32) * abs_vals).astype(np.float16)
            except struct.error:
                pass

        # Absolute mode
        vmin, scale, num_levels = struct.unpack_from("<ffi", payload)
        if num_levels == 0:
            return np.full(shape, vmin, dtype=np.float16)
        quantized = np.frombuffer(payload[header_size:], dtype=np.uint16).reshape(shape)
        return (quantized.astype(np.float32) * scale + vmin).astype(np.float16)

    def _compress_bytes(self, data: bytes) -> bytes:
        """Compress raw bytes with zstd or zlib."""
        if self._zstd_compressor is not None:
            return self._zstd_compressor.compress(data)
        return zlib.compress(data, level=6)

    def _decompress_bytes(self, data: bytes) -> bytes:
        """Decompress raw bytes."""
        if self._zstd_decompressor is not None:
            return self._zstd_decompressor.decompress(data)
        return zlib.decompress(data)

    @staticmethod
    def _serialize_rng_state(state: Dict[str, Any]) -> bytes:
        """Serialize RNG state dict to bytes."""
        import pickle
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize_rng_state(data: bytes) -> Dict[str, Any]:
        """Deserialize RNG state from bytes."""
        import pickle
        return pickle.loads(data)

    # ─── Statistics ───────────────────────────────────────────────────────────

    @staticmethod
    def compute_compression_stats(
        original: TrajectoryRecord,
        compressed: CompressedTrajectory,
    ) -> Dict[str, Any]:
        """Compute detailed compression statistics."""
        step_sizes_orig = [s.size_bytes() for s in original.steps]
        step_sizes_comp = [s.size_bytes() for s in compressed.steps]

        return {
            "original_total_bytes": compressed.original_size_bytes,
            "compressed_total_bytes": compressed.compressed_size_bytes,
            "compression_ratio": compressed.compression_ratio,
            "num_steps": len(compressed.steps),
            "avg_step_original_bytes": (
                np.mean(step_sizes_orig) if step_sizes_orig else 0
            ),
            "avg_step_compressed_bytes": (
                np.mean(step_sizes_comp) if step_sizes_comp else 0
            ),
            "per_step_ratios": [
                o / c if c > 0 else 0
                for o, c in zip(step_sizes_orig, step_sizes_comp)
            ],
        }
