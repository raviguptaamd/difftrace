#!/usr/bin/env python3
"""
DiffTrace Real Model Benchmark: LLaDA-8B-Instruct on AMD MI300X.

Measures actual provenance overhead against real dLLM inference.
This is the gold-standard benchmark for the paper.

Usage:
    python3 bench_real_llada.py --model-path /shared_inference/models/LLaDA-8B-Instruct
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
import torch.nn.functional as F

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
from difftrace.core.trajectory import TrajectoryRecord, DenoiseStep
from difftrace.core.compressor import DifferentialCompressor, CompressedTrajectory
from difftrace.core.async_io import AsyncProvenanceWriter
from difftrace.core.store import ProvenanceStore
from difftrace.core.replay import ReplayEngine
from difftrace.engine import DiffTraceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("bench_real_llada")

MASK_ID = 126336  # LLaDA's [MASK] token ID


# ─── LLaDA generation with DiffTrace hooks ──────────────────────────────────

def add_gumbel_noise(logits, temperature):
    """Gumbel max sampling for MDMs (from LLaDA generate.py)."""
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Compute tokens to unmask per step (linear schedule)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def generate_with_difftrace(
    model,
    tokenizer,
    prompt_text: str,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    device: str = "cuda:0",
    capture_config: Optional[CaptureGranularity] = None,
    store_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run LLaDA generation with full DiffTrace provenance capture.
    
    Returns dict with timing info, generated text, and trajectory.
    """
    # Prepare input
    messages = [{"role": "user", "content": prompt_text}]
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    prompt_length = input_ids.shape[1]

    # Initialize sequence with masks
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.clone()
    prompt_index = (x != MASK_ID)

    # DiffTrace trajectory
    trajectory = TrajectoryRecord(
        request_id=f"llada_{time.time_ns()}",
        prompt_tokens=input_ids[0].cpu().numpy().astype(np.int32),
        model_name="LLaDA-8B-Instruct",
        num_diffusion_steps=steps,
        sequence_length=prompt_length + gen_length,
        device=device,
        world_size=1,
        rank=0,
        start_time_ns=time.time_ns(),
    )

    # Block-based semi-autoregressive generation
    num_blocks = (gen_length + block_length - 1) // block_length
    steps_per_block = max(steps // num_blocks, 1)

    step_index = 0
    total_model_time_ns = 0
    total_capture_time_ns = 0

    torch.cuda.synchronize()
    gen_start = time.perf_counter_ns()

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])

        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        if not block_mask_index.any():
            continue

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == MASK_ID)
            if not mask_index.any():
                break

            # ── Model forward pass (the expensive part) ──
            torch.cuda.synchronize()
            t_model_start = time.perf_counter_ns()

            with torch.no_grad():
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = MASK_ID
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

            torch.cuda.synchronize()
            t_model_end = time.perf_counter_ns()
            total_model_time_ns += (t_model_end - t_model_start)

            # ── Token selection ──
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            x0_p[:, block_end:] = -float("inf")

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float("inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                block_confidence = confidence[j, block_start:block_end]
                if i < steps_per_block - 1:
                    k = min(num_transfer_tokens[j, i].item(), block_confidence.numel())
                    _, select_indices = torch.topk(block_confidence, k=k)
                    select_indices = select_indices + block_start
                    transfer_index[j, select_indices] = True
                else:
                    transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]

            x = torch.where(transfer_index, x0, x)

            # ── DiffTrace capture (the thing we're measuring) ──
            torch.cuda.synchronize()
            t_capture_start = time.perf_counter_ns()

            if capture_config is not None:
                # Capture unmasked positions and tokens for this step
                newly_unmasked = transfer_index[0].cpu().numpy()
                positions = np.where(newly_unmasked)[0].astype(np.int32)
                tokens = x0[0, positions].cpu().numpy().astype(np.int32) if len(positions) > 0 else np.array([], dtype=np.int32)
                
                mask_state = None
                step_logits = None
                rng_state = None

                if capture_config in (CaptureGranularity.TOKENS_AND_MASKS, CaptureGranularity.FULL_LOGITS):
                    current_mask = (x[0] == MASK_ID).cpu().numpy()
                    mask_state = current_mask

                if capture_config == CaptureGranularity.FULL_LOGITS:
                    step_logits = logits[0].cpu().to(torch.float16).numpy()

                ds = DenoiseStep(
                    step_index=step_index,
                    timestamp_ns=time.time_ns(),
                    unmasked_positions=positions,
                    sampled_tokens=tokens,
                    mask_state=mask_state,
                    logits=step_logits,
                    num_unmasked=int(newly_unmasked.sum()),
                    rng_state=rng_state,
                )
                trajectory.steps.append(ds)

            torch.cuda.synchronize()
            t_capture_end = time.perf_counter_ns()
            total_capture_time_ns += (t_capture_end - t_capture_start)

            step_index += 1

    torch.cuda.synchronize()
    gen_end = time.perf_counter_ns()

    trajectory.generated_tokens = x[0].cpu().numpy().astype(np.int32)
    trajectory.end_time_ns = time.time_ns()

    # Decode output
    response_tokens = x[0, prompt_length:]
    output_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    total_gen_ms = (gen_end - gen_start) / 1e6
    model_ms = total_model_time_ns / 1e6
    capture_ms = total_capture_time_ns / 1e6

    return {
        "output_text": output_text,
        "trajectory": trajectory,
        "total_gen_ms": total_gen_ms,
        "model_ms": model_ms,
        "capture_ms": capture_ms,
        "overhead_pct": (capture_ms / model_ms * 100) if model_ms > 0 else 0,
        "prompt_length": prompt_length,
        "gen_length": gen_length,
        "steps": steps,
        "actual_steps": step_index,
    }


# ─── Benchmark runner ────────────────────────────────────────────────────────

PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a short poem about the ocean at sunset.",
    "What are the main differences between Python and Rust?",
    "Describe the process of photosynthesis step by step.",
    "What is the significance of the Higgs boson discovery?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "How does a neural network learn from data?",
    "What are the advantages of renewable energy sources?",
    "Explain quantum entanglement to a high school student.",
    "Write a haiku about artificial intelligence.",
]


def run_benchmark(
    model,
    tokenizer,
    device: str,
    gen_lengths: List[int],
    step_counts: List[int],
    num_warmup: int = 2,
    num_runs: int = 5,
    store_dir: str = "/tmp/difftrace_real_bench",
) -> Dict[str, Any]:
    """Run the full benchmark suite."""

    results = []

    configs = {
        "baseline": None,  # No capture
        "tokens_only": CaptureGranularity.TOKENS_ONLY,
        "tokens_masks": CaptureGranularity.TOKENS_AND_MASKS,
        # "full_logits": CaptureGranularity.FULL_LOGITS,  # Too slow for systematic bench
    }

    for gen_length in gen_lengths:
        for steps in step_counts:
            for config_name, capture_gran in configs.items():
                logger.info(f"\n--- {config_name}: gen_len={gen_length}, steps={steps} ---")

                # Warmup
                for w in range(num_warmup):
                    prompt = PROMPTS[w % len(PROMPTS)]
                    _ = generate_with_difftrace(
                        model, tokenizer, prompt,
                        gen_length=gen_length, steps=steps,
                        device=device, capture_config=capture_gran,
                    )
                    torch.cuda.empty_cache()

                # Benchmark runs
                run_data = []
                for r in range(num_runs):
                    prompt = PROMPTS[r % len(PROMPTS)]
                    torch.cuda.synchronize()

                    result = generate_with_difftrace(
                        model, tokenizer, prompt,
                        gen_length=gen_length, steps=steps,
                        device=device, capture_config=capture_gran,
                    )

                    run_data.append({
                        "total_ms": result["total_gen_ms"],
                        "model_ms": result["model_ms"],
                        "capture_ms": result["capture_ms"],
                        "overhead_pct": result["overhead_pct"],
                        "actual_steps": result["actual_steps"],
                    })

                    if r == 0:
                        logger.info(f"  Output: {result['output_text'][:80]}...")

                    torch.cuda.empty_cache()

                # Aggregate
                avg_total = np.mean([d["total_ms"] for d in run_data])
                avg_model = np.mean([d["model_ms"] for d in run_data])
                avg_capture = np.mean([d["capture_ms"] for d in run_data])
                avg_overhead = np.mean([d["overhead_pct"] for d in run_data])
                std_total = np.std([d["total_ms"] for d in run_data])

                logger.info(
                    f"  avg_total={avg_total:.1f}ms, model={avg_model:.1f}ms, "
                    f"capture={avg_capture:.2f}ms, overhead={avg_overhead:.2f}%"
                )

                result_entry = {
                    "config": config_name,
                    "gen_length": gen_length,
                    "steps": steps,
                    "num_runs": num_runs,
                    "avg_total_ms": avg_total,
                    "std_total_ms": std_total,
                    "avg_model_ms": avg_model,
                    "avg_capture_ms": avg_capture,
                    "avg_overhead_pct": avg_overhead,
                    "actual_steps": run_data[0]["actual_steps"],
                    "runs": run_data,
                }
                results.append(result_entry)

    # Also do a compression + replay test with a real trajectory
    logger.info("\n=== Compression + Replay Test (real trajectory) ===")
    real_result = generate_with_difftrace(
        model, tokenizer, PROMPTS[0],
        gen_length=128, steps=64,
        device=device, capture_config=CaptureGranularity.TOKENS_AND_MASKS,
    )
    traj = real_result["trajectory"]

    comp_results = {}
    for mode_name, mode in [("none", CompressionMode.NONE), 
                              ("diff_lossless", CompressionMode.DIFF_LOSSLESS),
                              ("diff_lossy", CompressionMode.DIFF_LOSSY)]:
        comp_config = CompressionConfig(mode=mode)
        compressor = DifferentialCompressor(comp_config)
        
        t0 = time.perf_counter_ns()
        compressed = compressor.compress_trajectory(traj)
        t1 = time.perf_counter_ns()
        
        orig_size = compressed.original_size_bytes
        comp_size = compressed.compressed_size_bytes
        ratio = orig_size / comp_size if comp_size > 0 else 0
        
        comp_results[mode_name] = {
            "original_bytes": orig_size,
            "compressed_bytes": comp_size,
            "ratio": ratio,
            "compress_ms": (t1 - t0) / 1e6,
        }
        logger.info(f"  {mode_name}: {ratio:.1f}x ({orig_size} -> {comp_size} bytes, {(t1-t0)/1e6:.1f}ms)")

    # Replay test — compress first, then replay from compressed
    comp_config_replay = CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS)
    compressor_replay = DifferentialCompressor(comp_config_replay)
    compressed_for_replay = compressor_replay.compress_trajectory(traj)
    
    replay_engine = ReplayEngine(compressor=compressor_replay)
    t0 = time.perf_counter_ns()
    replay_result = replay_engine.token_replay(compressed_for_replay)
    t1 = time.perf_counter_ns()
    replay_ms = (t1 - t0) / 1e6
    match_rate = replay_result.token_match_rate if hasattr(replay_result, 'token_match_rate') else 0.0
    mismatches = replay_result.mismatched_positions if hasattr(replay_result, 'mismatched_positions') else []
    logger.info(f"  Replay: match_rate={match_rate:.0%}, time={replay_ms:.3f}ms")

    return {
        "benchmark": results,
        "compression_test": comp_results,
        "replay_test": {
            "match_rate": match_rate,
            "replay_ms": replay_ms,
            "num_mismatches": len(mismatches) if mismatches else 0,
        },
        "sample_output": real_result["output_text"],
    }


def main():
    parser = argparse.ArgumentParser(description="DiffTrace Real LLaDA Benchmark")
    parser.add_argument("--model-path", default="/shared_inference/models/LLaDA-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/real_llada.json")
    parser.add_argument("--gen-lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--steps", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-runs", type=int, default=5)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DiffTrace Real LLaDA-8B-Instruct Benchmark")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {args.device}")

    # Load model
    logger.info("Loading LLaDA-8B-Instruct...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()

    gpu_name = torch.cuda.get_device_name(args.device)
    gpu_mem = torch.cuda.get_device_properties(args.device).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    logger.info(f"Model loaded. GPU memory: {torch.cuda.memory_allocated(args.device)/1e9:.1f} GB")

    # Quick sanity check
    logger.info("\nSanity check: generating a short response...")
    test = generate_with_difftrace(
        model, tokenizer, "Hello, what is 2+2?",
        gen_length=32, steps=16, device=args.device,
        capture_config=CaptureGranularity.TOKENS_AND_MASKS,
    )
    logger.info(f"  Response: {test['output_text']}")
    logger.info(f"  Time: {test['total_gen_ms']:.1f}ms, Model: {test['model_ms']:.1f}ms")

    # Run full benchmark
    results = run_benchmark(
        model, tokenizer, args.device,
        gen_lengths=args.gen_lengths,
        step_counts=args.steps,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    # Add metadata
    output = {
        "experiment": "real_llada_overhead",
        "model": "LLaDA-8B-Instruct",
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem,
        "model_memory_gb": torch.cuda.memory_allocated(args.device) / 1e9,
        "torch_version": torch.__version__,
        "results": results["benchmark"],
        "compression_test": results["compression_test"],
        "replay_test": results["replay_test"],
        "sample_output": results["sample_output"],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: DiffTrace Overhead on Real LLaDA-8B-Instruct (MI300X)")
    print("=" * 70)
    print(f"{'Config':<15} {'GenLen':<8} {'Steps':<6} {'Total(ms)':<12} {'Model(ms)':<12} {'Capture(ms)':<12} {'Overhead%':<10}")
    print("-" * 75)
    for r in results["benchmark"]:
        print(
            f"{r['config']:<15} {r['gen_length']:<8} {r['steps']:<6} "
            f"{r['avg_total_ms']:<12.1f} {r['avg_model_ms']:<12.1f} "
            f"{r['avg_capture_ms']:<12.2f} {r['avg_overhead_pct']:<10.2f}"
        )

    print(f"\nCompression (real trajectory, gen_len=128, steps=64):")
    for name, cr in results["compression_test"].items():
        print(f"  {name}: {cr['ratio']:.1f}x ({cr['original_bytes']} -> {cr['compressed_bytes']} bytes)")

    mr = results['replay_test']['match_rate']
    rms = results['replay_test']['replay_ms']
    print(f"\nReplay: {mr:.0%} token match, {rms:.3f}ms")
    print(f"\nSample output: {results['sample_output'][:100]}...")


if __name__ == "__main__":
    main()
