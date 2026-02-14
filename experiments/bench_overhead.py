#!/usr/bin/env python3
"""
Experiment 1: DiffTrace Overhead Measurement.

Measures the wall-clock latency overhead of DiffTrace provenance capture
at different capture granularities compared to baseline (no provenance) inference.

Metrics:
- End-to-end inference latency (ms)
- Per-step capture overhead (ms)
- GPU memory overhead (MB)
- DiffTrace overhead as % of baseline

Configurations:
- Baseline: No DiffTrace
- Minimal: tokens_only capture, async I/O
- Medium: tokens_and_masks capture, async I/O
- Full: full_logits capture, async I/O
- Sync: full_logits capture, sync I/O (worst case)
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

# Add project root to path
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
from difftrace.hooks.llada import LLaDADiffTraceHook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("bench_overhead")


def get_configs() -> Dict[str, DiffTraceConfig]:
    """Define experiment configurations."""
    base_store = StoreConfig(base_path=Path("/tmp/difftrace_bench_overhead"))

    return {
        "baseline": DiffTraceConfig(enabled=False, store=base_store),
        "minimal": DiffTraceConfig(
            capture=CaptureConfig(granularity=CaptureGranularity.TOKENS_ONLY),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS),
            io=IOConfig(mode=IOMode.ASYNC),
            store=StoreConfig(base_path=base_store.base_path / "minimal"),
            enabled=True,
        ),
        "medium": DiffTraceConfig(
            capture=CaptureConfig(granularity=CaptureGranularity.TOKENS_AND_MASKS),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS),
            io=IOConfig(mode=IOMode.ASYNC),
            store=StoreConfig(base_path=base_store.base_path / "medium"),
            enabled=True,
        ),
        "full_async": DiffTraceConfig(
            capture=CaptureConfig(granularity=CaptureGranularity.FULL_LOGITS),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSY),
            io=IOConfig(mode=IOMode.ASYNC),
            store=StoreConfig(base_path=base_store.base_path / "full_async"),
            enabled=True,
        ),
        "full_sync": DiffTraceConfig(
            capture=CaptureConfig(granularity=CaptureGranularity.FULL_LOGITS),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSY),
            io=IOConfig(mode=IOMode.SYNC),
            store=StoreConfig(base_path=base_store.base_path / "full_sync"),
            enabled=True,
        ),
    }


def create_mock_model(vocab_size: int = 32000, hidden_dim: int = 4096, device: str = "cuda:0"):
    """
    Create a lightweight mock dLLM model for benchmarking.
    This simulates the compute characteristics of LLaDA without
    loading the full 8B model, enabling overhead measurement.
    """
    class MockDLLMOutput:
        def __init__(self, logits):
            self.logits = logits

    class MockDLLM(torch.nn.Module):
        def __init__(self, vocab_size, hidden_dim, device):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.embed = torch.nn.Embedding(vocab_size + 1, hidden_dim, device=device)  # +1 for mask token
            self.proj = torch.nn.Linear(hidden_dim, vocab_size + 1, device=device)
            self.name_or_path = "mock-dllm"
            self._device = device

        def forward(self, x):
            h = self.embed(x.clamp(0, self.vocab_size))
            logits = self.proj(h)
            return MockDLLMOutput(logits)

    return MockDLLM(vocab_size, hidden_dim, device).to(device)


def run_benchmark(
    model,
    config: DiffTraceConfig,
    config_name: str,
    seq_lengths: List[int],
    step_counts: List[int],
    num_warmup: int = 2,
    num_runs: int = 5,
    device: str = "cuda:0",
) -> List[Dict[str, Any]]:
    """Run benchmark for a single configuration."""
    results = []

    for seq_len in seq_lengths:
        for steps in step_counts:
            logger.info(f"  Config={config_name}, seq_len={seq_len}, steps={steps}")

            hook = LLaDADiffTraceHook(
                config=config,
                mask_token_id=32000,
                store=None,  # Don't store for overhead measurement
            )

            # Create dummy prompt
            prompt_len = min(32, seq_len // 4)
            prompt_ids = torch.randint(0, 32000, (1, prompt_len), device=device)

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    hook.generate(
                        model=model,
                        prompt_ids=prompt_ids,
                        seq_length=seq_len,
                        steps=steps,
                        device=device,
                    )
                torch.cuda.synchronize()

            # Benchmark runs
            latencies_ms = []
            mem_before = torch.cuda.memory_allocated(device) / 1e6
            peak_mem = 0

            for run_idx in range(num_runs):
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()

                t_start = time.perf_counter_ns()
                with torch.no_grad():
                    output, traj = hook.generate(
                        model=model,
                        prompt_ids=prompt_ids,
                        seq_length=seq_len,
                        steps=steps,
                        device=device,
                    )
                torch.cuda.synchronize()
                t_end = time.perf_counter_ns()

                latency_ms = (t_end - t_start) / 1e6
                latencies_ms.append(latency_ms)
                peak_mem = max(
                    peak_mem,
                    torch.cuda.max_memory_allocated(device) / 1e6,
                )

            mem_after = torch.cuda.memory_allocated(device) / 1e6

            result = {
                "config": config_name,
                "seq_length": seq_len,
                "steps": steps,
                "prompt_length": prompt_len,
                "avg_latency_ms": float(np.mean(latencies_ms)),
                "std_latency_ms": float(np.std(latencies_ms)),
                "min_latency_ms": float(np.min(latencies_ms)),
                "max_latency_ms": float(np.max(latencies_ms)),
                "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
                "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
                "peak_memory_mb": float(peak_mem),
                "memory_overhead_mb": float(mem_after - mem_before),
                "capture_overhead_ms": float(hook.avg_overhead_ms),
                "num_runs": num_runs,
            }
            results.append(result)
            logger.info(
                f"    avg_latency={result['avg_latency_ms']:.1f}ms, "
                f"peak_mem={result['peak_memory_mb']:.0f}MB, "
                f"capture_overhead={result['capture_overhead_ms']:.2f}ms"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="DiffTrace Overhead Benchmark")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--output", default="results/overhead.json", help="Output file")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--use-real-model", action="store_true", help="Use real LLaDA model")
    parser.add_argument("--model-path", default="GSAI/LLaDA-8B-Instruct")
    args = parser.parse_args()

    # Setup
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = args.device
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")

    # Create or load model
    if args.use_real_model:
        logger.info(f"Loading real model from {args.model_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device)
    else:
        logger.info("Using mock dLLM model for overhead benchmarking")
        model = create_mock_model(args.vocab_size, args.hidden_dim, device)

    # Run benchmarks
    configs = get_configs()
    all_results = []

    for config_name, config in configs.items():
        logger.info(f"\n=== Benchmarking config: {config_name} ===")
        results = run_benchmark(
            model=model,
            config=config,
            config_name=config_name,
            seq_lengths=args.seq_lengths,
            step_counts=args.steps,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            device=device,
        )
        all_results.extend(results)

    # Compute overhead percentages relative to baseline
    baseline_latencies = {
        (r["seq_length"], r["steps"]): r["avg_latency_ms"]
        for r in all_results if r["config"] == "baseline"
    }
    for r in all_results:
        key = (r["seq_length"], r["steps"])
        baseline = baseline_latencies.get(key, 0)
        if baseline > 0:
            r["overhead_pct"] = ((r["avg_latency_ms"] - baseline) / baseline) * 100
        else:
            r["overhead_pct"] = 0.0

    # Save results
    output = {
        "experiment": "overhead",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        "model": "mock-dllm" if not args.use_real_model else args.model_path,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n=== Overhead Summary ===")
    print(f"{'Config':<15} {'SeqLen':<8} {'Steps':<6} {'Latency(ms)':<14} {'Overhead%':<10}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['config']:<15} {r['seq_length']:<8} {r['steps']:<6} "
            f"{r['avg_latency_ms']:<14.1f} {r['overhead_pct']:<10.1f}"
        )


if __name__ == "__main__":
    main()
