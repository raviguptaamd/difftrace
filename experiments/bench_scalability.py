#!/usr/bin/env python3
"""
Experiment 4: Multi-Node Multi-GPU Scalability Benchmark.

Evaluates DiffTrace's overhead and provenance throughput as inference
scales across multiple MI300X nodes and GPUs using PyTorch distributed.

Metrics:
- Overhead % vs number of GPUs/nodes
- Provenance I/O throughput at scale
- Per-GPU memory overhead
- Distributed coordination overhead (including cross-node)
- Aggregate compression ratio

Single-node launch:
    torchrun --nproc_per_node=8 bench_scalability.py

Multi-node launch (via Slurm):
    srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
         --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         bench_scalability.py
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
import torch.distributed as dist

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
    DistributedConfig,
)
from difftrace.core.trajectory import TrajectoryRecord, DenoiseStep
from difftrace.core.compressor import DifferentialCompressor
from difftrace.core.distributed import DistributedCoordinator
from difftrace.engine import DiffTraceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("bench_scalability")


def setup_distributed():
    """Initialize PyTorch distributed."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def generate_per_rank_trajectory(
    rank: int,
    world_size: int,
    seq_length: int = 512,
    num_steps: int = 64,
    vocab_size: int = 32000,
    batch_size: int = 1,
) -> TrajectoryRecord:
    """
    Generate a synthetic trajectory simulating per-rank inference.

    In tensor-parallel inference, each rank processes a shard of the model.
    The trajectory structure is the same but token predictions may differ.
    """
    np.random.seed(42 + rank)
    prompt_length = 32
    gen_length = seq_length - prompt_length

    prompt_tokens = np.random.randint(0, vocab_size, size=prompt_length, dtype=np.int32)
    is_masked = np.ones(gen_length, dtype=bool)
    all_tokens = np.zeros(gen_length, dtype=np.int32)

    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    trajectory = TrajectoryRecord(
        request_id=f"scale_test_r{rank}_w{world_size}_{seq_length}",
        prompt_tokens=prompt_tokens,
        model_name=f"test-dllm-rank{rank}",
        num_diffusion_steps=num_steps,
        sequence_length=seq_length,
        device=f"cuda:{local_rank}",
        world_size=world_size,
        rank=rank,
        start_time_ns=time.time_ns(),
    )

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

        ds = DenoiseStep(
            step_index=step,
            timestamp_ns=time.time_ns(),
            unmasked_positions=(selected + prompt_length).astype(np.int32),
            sampled_tokens=tokens,
            mask_state=np.concatenate([np.zeros(prompt_length, dtype=bool), is_masked.copy()]),
            num_unmasked=num_to_unmask,
        )
        trajectory.steps.append(ds)

    trajectory.generated_tokens = np.concatenate([prompt_tokens, all_tokens])
    trajectory.end_time_ns = time.time_ns()
    return trajectory


def simulate_inference_gpu(
    seq_length: int,
    num_steps: int,
    local_rank: int,
    hidden_dim: int = 4096,
    vocab_size: int = 32000,
):
    """
    Simulate dLLM inference GPU workload via representative matmuls.
    Each denoising step involves a forward pass ~ seq_len x hidden x vocab matmul.
    This gives a realistic baseline for overhead calculation.
    """
    device = f"cuda:{local_rank}"
    x = torch.randn(seq_length, hidden_dim, device=device, dtype=torch.float16)
    w = torch.randn(hidden_dim, vocab_size, device=device, dtype=torch.float16)
    for _ in range(num_steps):
        _ = torch.mm(x, w)  # Simulate forward pass
    torch.cuda.synchronize()


def run_scalability_benchmark(
    rank: int,
    world_size: int,
    local_rank: int,
    seq_lengths: List[int],
    step_counts: List[int],
    num_requests: int = 10,
    vocab_size: int = 32000,
    output_dir: str = "/tmp/difftrace_scalability",
) -> List[Dict[str, Any]]:
    """Run scalability benchmark on this rank."""
    results = []

    for seq_len in seq_lengths:
        for num_steps in step_counts:
            if rank == 0:
                logger.info(f"\n--- seq_len={seq_len}, steps={num_steps}, world_size={world_size} ---")

            # Setup DiffTrace engine with distributed config
            config = DiffTraceConfig(
                capture=CaptureConfig(granularity=CaptureGranularity.TOKENS_AND_MASKS),
                compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS),
                io=IOConfig(mode=IOMode.ASYNC),
                store=StoreConfig(
                    base_path=Path(output_dir) / f"w{world_size}_s{seq_len}_t{num_steps}"
                ),
                distributed=DistributedConfig(
                    enabled=True,
                    world_size=world_size,
                    rank=rank,
                    backend="nccl",
                ),
                enabled=True,
            )

            engine = DiffTraceEngine(config)

            # Warmup (including GPU warmup)
            for _ in range(2):
                simulate_inference_gpu(seq_len, num_steps, local_rank, vocab_size=vocab_size)
                traj = generate_per_rank_trajectory(
                    rank, world_size, seq_len, num_steps, vocab_size
                )
                engine.store_trajectory(traj)

            dist.barrier()

            # === Baseline: simulate inference WITHOUT DiffTrace ===
            torch.cuda.synchronize()
            t_base_start = time.perf_counter_ns()
            for req_idx in range(num_requests):
                simulate_inference_gpu(seq_len, num_steps, local_rank, vocab_size=vocab_size)
                # Generate trajectory but don't store (simulates inference output)
                traj = generate_per_rank_trajectory(
                    rank, world_size, seq_len, num_steps, vocab_size
                )
            dist.barrier()
            torch.cuda.synchronize()
            t_base_end = time.perf_counter_ns()
            baseline_time_ms = (t_base_end - t_base_start) / 1e6

            dist.barrier()

            # === DiffTrace: simulate inference WITH provenance capture ===
            torch.cuda.synchronize()
            t_start = time.perf_counter_ns()

            total_bytes_original = 0
            total_bytes_compressed = 0

            for req_idx in range(num_requests):
                simulate_inference_gpu(seq_len, num_steps, local_rank, vocab_size=vocab_size)
                traj = generate_per_rank_trajectory(
                    rank, world_size, seq_len, num_steps, vocab_size
                )
                traj.request_id = f"bench_r{rank}_w{world_size}_{seq_len}_{num_steps}_{req_idx}"
                record = engine.store_trajectory(traj)
                if record:
                    total_bytes_original += record.original_size_bytes
                    total_bytes_compressed += record.compressed_size_bytes

            engine.flush()
            dist.barrier()

            torch.cuda.synchronize()
            t_end = time.perf_counter_ns()

            total_time_ms = (t_end - t_start) / 1e6
            time_per_request_ms = total_time_ms / num_requests

            # GPU memory
            mem_allocated = torch.cuda.memory_allocated(local_rank) / 1e6
            mem_reserved = torch.cuda.memory_reserved(local_rank) / 1e6

            stats = engine.get_stats()
            engine.shutdown()

            result = {
                "rank": rank,
                "world_size": world_size,
                "seq_length": seq_len,
                "num_steps": num_steps,
                "num_requests": num_requests,
                "total_time_ms": total_time_ms,
                "time_per_request_ms": time_per_request_ms,
                "baseline_time_ms": baseline_time_ms,
                "baseline_per_request_ms": baseline_time_ms / num_requests,
                "overhead_pct": (
                    (total_time_ms - baseline_time_ms) / baseline_time_ms * 100
                    if baseline_time_ms > 0 else 0
                ),
                "total_original_bytes": total_bytes_original,
                "total_compressed_bytes": total_bytes_compressed,
                "compression_ratio": (
                    total_bytes_original / total_bytes_compressed
                    if total_bytes_compressed > 0 else 0
                ),
                "gpu_memory_allocated_mb": mem_allocated,
                "gpu_memory_reserved_mb": mem_reserved,
            }
            results.append(result)

            if rank == 0:
                logger.info(
                    f"  overhead={result['overhead_pct']:.1f}%, "
                    f"per_req={result['time_per_request_ms']:.1f}ms, "
                    f"ratio={result['compression_ratio']:.1f}x"
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="DiffTrace Scalability Benchmark")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--steps", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--store-dir", default="/tmp/difftrace_scalability")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    gpus_per_node = world_size // max(num_nodes, 1)

    if rank == 0:
        logger.info(f"=== DiffTrace Multi-Node Scalability Benchmark ===")
        logger.info(f"World size: {world_size} ({num_nodes} nodes x {gpus_per_node} GPUs/node)")
        logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        logger.info(f"Hostname: {os.environ.get('HOSTNAME', 'unknown')}")

    results = run_scalability_benchmark(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        seq_lengths=args.seq_lengths,
        step_counts=args.steps,
        num_requests=args.num_requests,
        vocab_size=args.vocab_size,
        output_dir=args.store_dir,
    )

    # Gather results at rank 0
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)

    if rank == 0:
        # Flatten and save
        flat_results = []
        for rank_results in all_results:
            flat_results.extend(rank_results)

        output = {
            "experiment": "scalability",
            "world_size": world_size,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "results": flat_results,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"scalability_w{world_size}.json")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to {output_file}")

        # Print summary (rank 0 only)
        print(f"\n=== Scalability Summary (world_size={world_size}) ===")
        rank0_results = [r for r in flat_results if r["rank"] == 0]
        print(f"{'SeqLen':<8} {'Steps':<6} {'Overhead%':<10} {'CompRatio':<10} {'PerReq(ms)':<12}")
        print("-" * 50)
        for r in rank0_results:
            print(
                f"{r['seq_length']:<8} {r['num_steps']:<6} "
                f"{r['overhead_pct']:<10.1f} {r['compression_ratio']:<10.1f} "
                f"{r['time_per_request_ms']:<12.1f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
