#!/usr/bin/env python3
"""
Experiment 2: Differential Compression Evaluation.

Evaluates the compression effectiveness of DiffTrace's differential encoding
compared to full-state storage.

Metrics:
- Compression ratio (original / compressed)
- Compression throughput (MB/s)
- Decompression throughput (MB/s)
- Per-step compression ratio evolution
- Storage per token generated (bytes/token)
- Lossy compression: reconstruction error (MSE, max error)

Configurations:
- None: No compression (raw storage)
- Diff+Lossless: Differential encoding + zstd
- Diff+Lossy: Differential encoding + error-bounded lossy for logits
- Varying error bounds for lossy compression
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from difftrace.utils.config import (
    DiffTraceConfig,
    CaptureConfig,
    CaptureGranularity,
    CompressionConfig,
    CompressionMode,
    IOConfig,
    IOMode,
    StoreConfig,
)
from difftrace.core.trajectory import TrajectoryCapture, TrajectoryRecord, DenoiseStep
from difftrace.core.compressor import DifferentialCompressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("bench_compression")


def generate_synthetic_trajectory(
    seq_length: int = 512,
    vocab_size: int = 32000,
    num_steps: int = 64,
    prompt_length: int = 32,
    include_logits: bool = False,
) -> TrajectoryRecord:
    """
    Generate a synthetic denoising trajectory that mimics LLaDA behavior.

    The trajectory simulates the masked diffusion process:
    - Start with all generation positions masked
    - Each step unmasks a fraction of remaining positions
    - Earlier steps have more diversity in sampled tokens
    - Later steps have higher confidence (less entropy)
    """
    gen_length = seq_length - prompt_length
    prompt_tokens = np.random.randint(0, vocab_size, size=prompt_length, dtype=np.int32)

    # Track which positions are still masked
    is_masked = np.ones(gen_length, dtype=bool)
    all_tokens = np.zeros(gen_length, dtype=np.int32)

    trajectory = TrajectoryRecord(
        request_id=f"synthetic_{seq_length}_{num_steps}",
        prompt_tokens=prompt_tokens,
        model_name="synthetic-dllm",
        num_diffusion_steps=num_steps,
        sequence_length=seq_length,
        start_time_ns=time.time_ns(),
    )

    for step in range(num_steps):
        t = num_steps - step
        num_remaining = is_masked.sum()
        if num_remaining == 0:
            break

        # Determine how many to unmask at this step
        fraction = 1.0 / max(t, 1)
        num_to_unmask = max(1, int(num_remaining * fraction))
        num_to_unmask = min(num_to_unmask, num_remaining)

        # Select positions to unmask (random among masked)
        masked_indices = np.where(is_masked)[0]
        selected = np.random.choice(masked_indices, size=num_to_unmask, replace=False)
        selected.sort()

        # Sample tokens for selected positions
        tokens = np.random.randint(0, vocab_size, size=num_to_unmask, dtype=np.int32)

        # Unmask
        is_masked[selected] = False
        all_tokens[selected] = tokens

        # Confidence: higher in later steps (lower noise)
        confidence = np.random.beta(2 + step * 0.5, 1, size=num_to_unmask).astype(np.float32)

        # Create DenoiseStep
        ds = DenoiseStep(
            step_index=step,
            timestamp_ns=time.time_ns(),
            unmasked_positions=(selected + prompt_length).astype(np.int32),
            sampled_tokens=tokens,
            mask_state=np.concatenate([
                np.zeros(prompt_length, dtype=bool),
                is_masked.copy(),
            ]),
            num_unmasked=num_to_unmask,
            confidence_scores=confidence,
        )

        # Optionally include logits
        if include_logits:
            # Simulate logits: sparse distribution (most mass on few tokens)
            logits = np.random.randn(seq_length, vocab_size).astype(np.float16)
            # Make it more realistic: sharper distribution
            logits *= (0.5 + step * 0.1)
            ds.logits = logits

        trajectory.steps.append(ds)

    trajectory.generated_tokens = np.concatenate([prompt_tokens, all_tokens])
    trajectory.end_time_ns = time.time_ns()
    return trajectory


def benchmark_compression(
    trajectory: TrajectoryRecord,
    config: CompressionConfig,
    config_name: str,
) -> Dict[str, Any]:
    """Benchmark compression for a single configuration."""
    compressor = DifferentialCompressor(config)

    # Measure compression time
    t_start = time.perf_counter_ns()
    compressed = compressor.compress_trajectory(trajectory)
    t_end = time.perf_counter_ns()
    compress_time_ms = (t_end - t_start) / 1e6

    # Compute detailed stats
    stats = DifferentialCompressor.compute_compression_stats(trajectory, compressed)

    original_mb = stats["original_total_bytes"] / 1e6
    compressed_mb = stats["compressed_total_bytes"] / 1e6
    throughput_mb_s = original_mb / (compress_time_ms / 1000) if compress_time_ms > 0 else 0

    # Per-token storage
    gen_length = trajectory.sequence_length - len(trajectory.prompt_tokens)
    bytes_per_token = stats["compressed_total_bytes"] / max(gen_length, 1)

    result = {
        "config": config_name,
        "seq_length": trajectory.sequence_length,
        "num_steps": trajectory.num_diffusion_steps,
        "num_steps_captured": len(trajectory.steps),
        "original_bytes": stats["original_total_bytes"],
        "compressed_bytes": stats["compressed_total_bytes"],
        "compression_ratio": stats["compression_ratio"],
        "compress_time_ms": compress_time_ms,
        "compress_throughput_mb_s": throughput_mb_s,
        "bytes_per_token": bytes_per_token,
        "avg_step_original_bytes": float(stats["avg_step_original_bytes"]),
        "avg_step_compressed_bytes": float(stats["avg_step_compressed_bytes"]),
    }

    # For lossy: measure reconstruction error
    if config.mode == CompressionMode.DIFF_LOSSY:
        errors = measure_lossy_error(trajectory, compressed, compressor)
        result.update(errors)

    return result


def measure_lossy_error(
    trajectory: TrajectoryRecord,
    compressed,
    compressor: DifferentialCompressor,
) -> Dict[str, float]:
    """Measure reconstruction error for lossy compression of logits."""
    max_errors = []
    mse_errors = []

    for orig_step, comp_step in zip(trajectory.steps, compressed.steps):
        if orig_step.logits is not None and comp_step.logits_data is not None:
            try:
                reconstructed = compressor.decompress_logits_lossy(
                    comp_step.logits_data,
                    shape=orig_step.logits.shape,
                )
                diff = np.abs(
                    orig_step.logits.astype(np.float32)
                    - reconstructed.astype(np.float32)
                )
                max_errors.append(float(diff.max()))
                mse_errors.append(float((diff ** 2).mean()))
            except Exception as e:
                logger.warning(f"Error measuring reconstruction: {e}")

    return {
        "lossy_max_error": float(np.max(max_errors)) if max_errors else 0.0,
        "lossy_avg_max_error": float(np.mean(max_errors)) if max_errors else 0.0,
        "lossy_avg_mse": float(np.mean(mse_errors)) if mse_errors else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="DiffTrace Compression Benchmark")
    parser.add_argument("--output", default="results/compression.json")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--error-bounds", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Compression configurations
    compression_configs = {
        "none": CompressionConfig(mode=CompressionMode.NONE),
        "diff_lossless": CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS, zstd_level=3),
        "diff_lossless_high": CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS, zstd_level=9),
    }

    # Add lossy configs with different error bounds
    for eb in args.error_bounds:
        name = f"diff_lossy_eb{eb}"
        compression_configs[name] = CompressionConfig(
            mode=CompressionMode.DIFF_LOSSY,
            zstd_level=3,
            lossy_error_bound=eb,
        )

    all_results = []

    for seq_len in args.seq_lengths:
        for num_steps in args.steps:
            logger.info(f"\n--- seq_len={seq_len}, steps={num_steps} ---")

            # Generate trajectories with and without logits
            traj_tokens = generate_synthetic_trajectory(
                seq_length=seq_len,
                vocab_size=args.vocab_size,
                num_steps=num_steps,
                include_logits=False,
            )
            traj_logits = generate_synthetic_trajectory(
                seq_length=seq_len,
                vocab_size=args.vocab_size,
                num_steps=num_steps,
                include_logits=True,
            )

            for config_name, config in compression_configs.items():
                # Use logits trajectory for lossy configs
                use_logits = "lossy" in config_name
                traj = traj_logits if use_logits else traj_tokens

                try:
                    result = benchmark_compression(traj, config, config_name)
                    all_results.append(result)
                    logger.info(
                        f"  {config_name}: ratio={result['compression_ratio']:.1f}x, "
                        f"time={result['compress_time_ms']:.1f}ms, "
                        f"bytes/token={result['bytes_per_token']:.1f}"
                    )
                except Exception as e:
                    logger.error(f"  {config_name}: FAILED - {e}")

    # Save results
    output = {
        "experiment": "compression",
        "vocab_size": args.vocab_size,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")

    # Summary
    print("\n=== Compression Summary ===")
    print(f"{'Config':<25} {'SeqLen':<8} {'Steps':<6} {'Ratio':<8} {'Time(ms)':<10} {'B/tok':<8}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['config']:<25} {r['seq_length']:<8} {r['num_steps']:<6} "
            f"{r['compression_ratio']:<8.1f} {r['compress_time_ms']:<10.1f} "
            f"{r['bytes_per_token']:<8.1f}"
        )


if __name__ == "__main__":
    main()
