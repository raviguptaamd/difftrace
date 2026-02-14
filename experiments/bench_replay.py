#!/usr/bin/env python3
"""
Experiment 3: Replay Accuracy and Performance.

Evaluates DiffTrace's ability to deterministically replay dLLM inference
from stored provenance records.

Metrics:
- Token-level match rate (% of tokens matching original)
- Exact match rate (% of sequences fully matching)
- Replay speedup over original inference
- First-divergence-step analysis (for non-exact matches)
- Replay time vs. original inference time

Scenarios:
- Same device, same random state -> should be exact match
- Different device -> may diverge (we measure where)
- Corrupted provenance -> graceful degradation
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

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
from difftrace.core.compressor import DifferentialCompressor, CompressedTrajectory
from difftrace.core.replay import ReplayEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("bench_replay")


def generate_realistic_trajectory(
    seq_length: int = 512,
    vocab_size: int = 32000,
    num_steps: int = 64,
    prompt_length: int = 32,
    seed: int = 42,
) -> TrajectoryRecord:
    """Generate a deterministic synthetic trajectory for replay testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    gen_length = seq_length - prompt_length
    prompt_tokens = np.random.randint(0, vocab_size, size=prompt_length, dtype=np.int32)

    is_masked = np.ones(gen_length, dtype=bool)
    all_tokens = np.zeros(gen_length, dtype=np.int32)

    trajectory = TrajectoryRecord(
        request_id=f"replay_test_{seq_length}_{num_steps}_s{seed}",
        prompt_tokens=prompt_tokens,
        model_name="test-dllm",
        num_diffusion_steps=num_steps,
        sequence_length=seq_length,
        temperature=1.0,
        start_time_ns=time.time_ns(),
    )

    # Simulate initial RNG capture
    trajectory.initial_rng_state = {
        "python_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state().numpy().tobytes(),
    }

    for step in range(num_steps):
        t = num_steps - step
        num_remaining = int(is_masked.sum())
        if num_remaining == 0:
            break

        fraction = 1.0 / max(t, 1)
        num_to_unmask = max(1, int(num_remaining * fraction))
        num_to_unmask = min(num_to_unmask, num_remaining)

        masked_indices = np.where(is_masked)[0]
        selected = np.sort(np.random.choice(masked_indices, size=num_to_unmask, replace=False))
        tokens = np.random.randint(0, vocab_size, size=num_to_unmask, dtype=np.int32)

        is_masked[selected] = False
        all_tokens[selected] = tokens

        full_positions = (selected + prompt_length).astype(np.int32)

        ds = DenoiseStep(
            step_index=step,
            timestamp_ns=time.time_ns(),
            unmasked_positions=full_positions,
            sampled_tokens=tokens,
            mask_state=np.concatenate([np.zeros(prompt_length, dtype=bool), is_masked.copy()]),
            num_unmasked=num_to_unmask,
        )
        trajectory.steps.append(ds)

    trajectory.generated_tokens = np.concatenate([prompt_tokens, all_tokens])
    trajectory.end_time_ns = time.time_ns()
    return trajectory


def run_replay_benchmark(
    seq_lengths: List[int],
    step_counts: List[int],
    num_trials: int = 10,
    vocab_size: int = 32000,
) -> List[Dict[str, Any]]:
    """Run replay benchmarks across configurations."""
    results = []
    compressor = DifferentialCompressor(
        CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS)
    )
    replay_engine = ReplayEngine(compressor)

    for seq_len in seq_lengths:
        for num_steps in step_counts:
            logger.info(f"\n--- seq_len={seq_len}, steps={num_steps} ---")

            exact_matches = 0
            token_match_rates = []
            replay_times_ms = []
            speedups = []

            for trial in range(num_trials):
                seed = trial * 1000 + seq_len

                # Generate trajectory
                traj = generate_realistic_trajectory(
                    seq_length=seq_len,
                    vocab_size=vocab_size,
                    num_steps=num_steps,
                    seed=seed,
                )

                # Compress
                compressed = compressor.compress_trajectory(traj)

                # Replay
                replay_result = replay_engine.token_replay(
                    compressed,
                    vocab_size=vocab_size,
                )

                if replay_result.success:
                    if replay_result.exact_match:
                        exact_matches += 1
                    token_match_rates.append(replay_result.token_match_rate)
                    replay_times_ms.append(replay_result.replay_time_ms)
                    if replay_result.speedup > 0:
                        speedups.append(replay_result.speedup)

            result = {
                "seq_length": seq_len,
                "num_steps": num_steps,
                "num_trials": num_trials,
                "exact_match_rate": exact_matches / num_trials,
                "avg_token_match_rate": float(np.mean(token_match_rates)) if token_match_rates else 0,
                "min_token_match_rate": float(np.min(token_match_rates)) if token_match_rates else 0,
                "avg_replay_time_ms": float(np.mean(replay_times_ms)) if replay_times_ms else 0,
                "std_replay_time_ms": float(np.std(replay_times_ms)) if replay_times_ms else 0,
                "avg_speedup": float(np.mean(speedups)) if speedups else 0,
            }
            results.append(result)

            logger.info(
                f"  exact_match={result['exact_match_rate']:.0%}, "
                f"token_match={result['avg_token_match_rate']:.2%}, "
                f"replay_time={result['avg_replay_time_ms']:.2f}ms"
            )

    return results


def run_divergence_analysis(
    num_trials: int = 50,
    seq_length: int = 512,
    num_steps: int = 64,
    vocab_size: int = 32000,
) -> Dict[str, Any]:
    """
    Analyze where two runs of the same inference diverge
    when random states are NOT restored (simulating non-reproducibility).
    """
    compressor = DifferentialCompressor(
        CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS)
    )
    replay_engine = ReplayEngine(compressor)

    first_divergence_steps = []

    for trial in range(num_trials):
        # Generate two trajectories with different seeds (simulating non-reproducibility)
        traj_a = generate_realistic_trajectory(
            seq_length=seq_length, num_steps=num_steps, seed=trial,
        )
        traj_b = generate_realistic_trajectory(
            seq_length=seq_length, num_steps=num_steps, seed=trial + 10000,
        )

        comp_a = compressor.compress_trajectory(traj_a)
        comp_b = compressor.compress_trajectory(traj_b)

        comparison = replay_engine.compare_trajectories(comp_a, comp_b)
        first_div = comparison["first_divergence_step"]
        if first_div >= 0:
            first_divergence_steps.append(first_div)

    return {
        "num_trials": num_trials,
        "divergence_rate": len(first_divergence_steps) / num_trials,
        "avg_first_divergence_step": float(np.mean(first_divergence_steps)) if first_divergence_steps else -1,
        "std_first_divergence_step": float(np.std(first_divergence_steps)) if first_divergence_steps else 0,
        "min_first_divergence_step": int(np.min(first_divergence_steps)) if first_divergence_steps else -1,
        "max_first_divergence_step": int(np.max(first_divergence_steps)) if first_divergence_steps else -1,
    }


def main():
    parser = argparse.ArgumentParser(description="DiffTrace Replay Benchmark")
    parser.add_argument("--output", default="results/replay.json")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("=== Replay Accuracy Benchmark ===")
    replay_results = run_replay_benchmark(
        seq_lengths=args.seq_lengths,
        step_counts=args.steps,
        num_trials=args.num_trials,
        vocab_size=args.vocab_size,
    )

    logger.info("\n=== Divergence Analysis ===")
    divergence_results = run_divergence_analysis(
        num_trials=50,
        seq_length=512,
        num_steps=64,
        vocab_size=args.vocab_size,
    )

    output = {
        "experiment": "replay",
        "replay_results": replay_results,
        "divergence_analysis": divergence_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")

    # Summary
    print("\n=== Replay Summary ===")
    print(f"{'SeqLen':<8} {'Steps':<6} {'ExactMatch':<12} {'TokenMatch':<12} {'ReplayMs':<10}")
    print("-" * 50)
    for r in replay_results:
        print(
            f"{r['seq_length']:<8} {r['num_steps']:<6} "
            f"{r['exact_match_rate']:<12.0%} {r['avg_token_match_rate']:<12.2%} "
            f"{r['avg_replay_time_ms']:<10.2f}"
        )

    print(f"\nDivergence Analysis:")
    print(f"  Divergence rate (no RNG restore): {divergence_results['divergence_rate']:.0%}")
    print(f"  Avg first divergence step: {divergence_results['avg_first_divergence_step']:.1f}")


if __name__ == "__main__":
    main()
