#!/usr/bin/env python3
"""
End-to-end test for DiffTrace.

Validates the full pipeline:
1. Synthetic trajectory generation
2. Compression (all modes)
3. Storage and retrieval
4. Token replay with verification
5. Comparison of trajectories

This test runs without GPU and without a real model,
ensuring the framework logic is correct.
"""

import os
import sys
import time
import shutil
import tempfile
import logging
from pathlib import Path

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
from difftrace.core.async_io import AsyncProvenanceWriter
from difftrace.core.store import ProvenanceStore
from difftrace.core.replay import ReplayEngine
from difftrace.engine import DiffTraceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_e2e")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        logger.info(f"  PASS: {name}")
    else:
        FAIL += 1
        logger.error(f"  FAIL: {name} - {detail}")


def generate_test_trajectory(
    seq_length: int = 256,
    vocab_size: int = 1000,
    num_steps: int = 32,
    prompt_length: int = 16,
    include_logits: bool = False,
    seed: int = 42,
) -> TrajectoryRecord:
    """Generate a deterministic test trajectory."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    gen_length = seq_length - prompt_length
    prompt_tokens = np.random.randint(0, vocab_size, size=prompt_length, dtype=np.int32)

    is_masked = np.ones(gen_length, dtype=bool)
    all_tokens = np.zeros(gen_length, dtype=np.int32)

    # Create trajectory with capture API
    config = CaptureConfig(
        granularity=CaptureGranularity.FULL_LOGITS if include_logits else CaptureGranularity.TOKENS_AND_MASKS,
        capture_random_state=False,  # Deterministic for testing
    )
    capture = TrajectoryCapture(config)

    trajectory = capture.begin_trajectory(
        request_id=f"test_{seq_length}_{num_steps}",
        prompt_tokens=prompt_tokens,
        prompt_text="This is a test prompt",
        model_name="test-model",
        num_diffusion_steps=num_steps,
        sequence_length=seq_length,
        device="cpu",
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

        full_positions = torch.from_numpy((selected + prompt_length).astype(np.int32))
        sampled_tokens = torch.from_numpy(tokens)

        mask_state = torch.from_numpy(np.concatenate([
            np.zeros(prompt_length, dtype=bool),
            is_masked.copy(),
        ]))

        logits = None
        if include_logits:
            logits = torch.randn(seq_length, vocab_size, dtype=torch.float16)

        capture.capture_step(
            step_index=step,
            unmasked_positions=full_positions,
            sampled_tokens=sampled_tokens,
            mask_state=mask_state,
            logits=logits,
            device="cpu",
        )

    generated = np.concatenate([prompt_tokens, all_tokens])
    trajectory = capture.end_trajectory(
        generated_tokens=torch.from_numpy(generated),
        generated_text="Test generated text output",
    )
    return trajectory


def test_trajectory_capture():
    """Test 1: Trajectory capture."""
    logger.info("\n=== Test 1: Trajectory Capture ===")

    traj = generate_test_trajectory()
    check("trajectory created", traj is not None)
    check("has steps", len(traj.steps) > 0, f"got {len(traj.steps)} steps")
    check("has prompt", len(traj.prompt_tokens) == 16)
    check("has generated", traj.generated_tokens is not None)
    check("sequence length matches", len(traj.generated_tokens) == 256)
    check("request_id set", traj.request_id == "test_256_32")
    check("timing recorded", traj.end_time_ns > traj.start_time_ns)

    # Verify step data
    for step in traj.steps:
        check(f"step {step.step_index} has positions",
              step.unmasked_positions is not None and len(step.unmasked_positions) > 0)
        check(f"step {step.step_index} has tokens",
              step.sampled_tokens is not None and len(step.sampled_tokens) > 0)
        check(f"step {step.step_index} has mask",
              step.mask_state is not None)
        break  # Just check first step

    return traj


def test_compression(traj: TrajectoryRecord):
    """Test 2: Differential compression."""
    logger.info("\n=== Test 2: Compression ===")

    # Test all compression modes
    for mode_name, mode in [
        ("none", CompressionMode.NONE),
        ("diff_lossless", CompressionMode.DIFF_LOSSLESS),
        ("diff_lossy", CompressionMode.DIFF_LOSSY),
    ]:
        # For lossy, use trajectory with logits
        if mode == CompressionMode.DIFF_LOSSY:
            test_traj = generate_test_trajectory(include_logits=True, seed=123)
        else:
            test_traj = traj

        config = CompressionConfig(mode=mode, zstd_level=3, lossy_error_bound=1e-2)
        compressor = DifferentialCompressor(config)

        compressed = compressor.compress_trajectory(test_traj)
        check(f"{mode_name}: compressed",
              compressed is not None,
              "compression returned None")
        check(f"{mode_name}: has steps",
              len(compressed.steps) == len(test_traj.steps),
              f"expected {len(test_traj.steps)}, got {len(compressed.steps)}")

        if mode != CompressionMode.NONE:
            ratio = compressed.compression_ratio
            check(f"{mode_name}: compression ratio > 1",
                  ratio > 1.0,
                  f"ratio = {ratio:.2f}")
            logger.info(f"    {mode_name}: ratio = {ratio:.1f}x, "
                       f"original = {compressed.original_size_bytes}, "
                       f"compressed = {compressed.compressed_size_bytes}")

    return compressed


def test_async_io():
    """Test 3: Async I/O engine."""
    logger.info("\n=== Test 3: Async I/O ===")

    tmpdir = tempfile.mkdtemp(prefix="difftrace_test_")
    try:
        # Test sync mode
        config_sync = IOConfig(mode=IOMode.SYNC)
        writer_sync = AsyncProvenanceWriter(config_sync, Path(tmpdir) / "sync")

        traj = generate_test_trajectory(seq_length=128, num_steps=8, seed=99)
        compressor = DifferentialCompressor(CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS))
        compressed = compressor.compress_trajectory(traj)

        success = writer_sync.write_trajectory(compressed)
        check("sync write succeeded", success)
        check("sync file exists",
              (Path(tmpdir) / "sync" / f"{traj.request_id}.dtrace").exists())

        stats = writer_sync.stats
        check("sync stats: 1 write", stats.total_writes == 1)

        # Test async mode
        config_async = IOConfig(mode=IOMode.ASYNC, flush_interval_ms=50)
        writer_async = AsyncProvenanceWriter(config_async, Path(tmpdir) / "async")

        for i in range(5):
            traj_i = generate_test_trajectory(seq_length=128, num_steps=8, seed=i)
            traj_i.request_id = f"async_test_{i}"
            comp_i = compressor.compress_trajectory(traj_i)
            writer_async.write_trajectory(comp_i)

        writer_async.flush_and_wait()
        time.sleep(0.5)  # Give workers time to write

        async_stats = writer_async.stats
        check("async: writes completed", async_stats.total_writes >= 1,
              f"got {async_stats.total_writes} writes")

        writer_async.shutdown()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_store():
    """Test 4: Provenance store."""
    logger.info("\n=== Test 4: Provenance Store ===")

    tmpdir = tempfile.mkdtemp(prefix="difftrace_test_store_")
    try:
        config = DiffTraceConfig(
            capture=CaptureConfig(granularity=CaptureGranularity.TOKENS_AND_MASKS),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS),
            io=IOConfig(mode=IOMode.SYNC),  # Sync for deterministic testing
            store=StoreConfig(base_path=Path(tmpdir)),
            enabled=True,
        )

        store = ProvenanceStore(config)

        # Store multiple trajectories
        for i in range(3):
            traj = generate_test_trajectory(seed=i * 100)
            traj.request_id = f"store_test_{i}"
            record = store.store_trajectory(traj)
            check(f"store record {i}", record is not None)

        # Query
        records = store.list_records()
        check("list returns 3 records", len(records) == 3, f"got {len(records)}")

        # Load back
        loaded = store.load_trajectory("store_test_0")
        check("load trajectory", loaded is not None)
        check("loaded has correct id", loaded.request_id == "store_test_0")

        # Stats
        stats = store.get_stats()
        check("stats has num_records", stats["num_records"] == 3)
        check("stats has compression ratio", stats["overall_compression_ratio"] > 0)

        store.shutdown()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_replay():
    """Test 5: Token replay."""
    logger.info("\n=== Test 5: Replay ===")

    traj = generate_test_trajectory(seq_length=256, num_steps=32, seed=42)

    compressor = DifferentialCompressor(
        CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS)
    )
    compressed = compressor.compress_trajectory(traj)

    replay_engine = ReplayEngine(compressor)
    result = replay_engine.token_replay(compressed)

    check("replay succeeded", result.success)
    check("exact match", result.exact_match,
          f"token_match_rate={result.token_match_rate:.4f}")
    check("match rate = 1.0", result.token_match_rate == 1.0)
    check("no mismatches", len(result.mismatched_positions) == 0,
          f"mismatches at {result.mismatched_positions[:5]}")
    check("replay is fast", result.replay_time_ms < 100,
          f"took {result.replay_time_ms:.1f}ms")

    logger.info(f"    Replay time: {result.replay_time_ms:.3f}ms")
    logger.info(f"    Speedup: {result.speedup:.0f}x")


def test_trajectory_comparison():
    """Test 6: Trajectory comparison."""
    logger.info("\n=== Test 6: Trajectory Comparison ===")

    traj_a = generate_test_trajectory(seed=42)
    traj_b = generate_test_trajectory(seed=42)  # Same seed = identical
    traj_c = generate_test_trajectory(seed=99)  # Different seed = divergent

    compressor = DifferentialCompressor(
        CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS)
    )
    replay_engine = ReplayEngine(compressor)

    comp_a = compressor.compress_trajectory(traj_a)
    comp_b = compressor.compress_trajectory(traj_b)
    comp_c = compressor.compress_trajectory(traj_c)

    # Same seed: should be identical
    result_same = replay_engine.compare_trajectories(comp_a, comp_b)
    check("same seed: no divergence",
          result_same["first_divergence_step"] == -1)

    # Different seed: should diverge
    result_diff = replay_engine.compare_trajectories(comp_a, comp_c)
    check("diff seed: diverges",
          result_diff["first_divergence_step"] >= 0,
          f"first_div={result_diff['first_divergence_step']}")


def test_engine():
    """Test 7: DiffTrace engine (high-level API)."""
    logger.info("\n=== Test 7: Engine ===")

    tmpdir = tempfile.mkdtemp(prefix="difftrace_test_engine_")
    try:
        config = DiffTraceConfig(
            capture=CaptureConfig(granularity=CaptureGranularity.TOKENS_AND_MASKS),
            compression=CompressionConfig(mode=CompressionMode.DIFF_LOSSLESS),
            io=IOConfig(mode=IOMode.SYNC),
            store=StoreConfig(base_path=Path(tmpdir)),
            enabled=True,
        )

        with DiffTraceEngine(config) as engine:
            # Store trajectory
            traj = generate_test_trajectory(seed=42)
            record = engine.store_trajectory(traj)
            check("engine store", record is not None)

            # List
            records = engine.list_records()
            check("engine list", len(records) == 1)

            # Replay
            result = engine.replay(traj.request_id)
            check("engine replay", result is not None and result.success)
            check("engine replay exact", result.exact_match)

            # Stats
            stats = engine.get_stats()
            check("engine stats", stats["enabled"] is True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    global PASS, FAIL

    logger.info("=" * 60)
    logger.info("DiffTrace End-to-End Test Suite")
    logger.info("=" * 60)

    traj = test_trajectory_capture()
    test_compression(traj)
    test_async_io()
    test_store()
    test_replay()
    test_trajectory_comparison()
    test_engine()

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {PASS} passed, {FAIL} failed")
    logger.info("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
