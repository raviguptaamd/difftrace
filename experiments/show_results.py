#!/usr/bin/env python3
"""Quick script to display experiment results."""
import json, os, sys, glob

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

# Scalability results
for f in sorted(glob.glob(os.path.join(results_dir, "scalability_w*.json"))):
    with open(f) as fh:
        data = json.load(fh)
    ws = data["world_size"]
    nn = data.get("num_nodes", "?")
    gpn = data.get("gpus_per_node", "?")
    gpu = data.get("gpu_name", "?")
    r0 = [r for r in data["results"] if r["rank"] == 0]
    print(f"\n=== {os.path.basename(f)}: {ws} GPUs ({nn} nodes x {gpn} GPUs) on {gpu} ===")
    hdr = f"{'SeqLen':<8} {'Steps':<6} {'Overhead%':<10} {'CompRatio':<10} {'ms/req':<10}"
    print(hdr)
    print("-" * len(hdr))
    for r in r0:
        print(f"{r['seq_length']:<8} {r['num_steps']:<6} {r['overhead_pct']:<10.1f} {r['compression_ratio']:<10.1f} {r['time_per_request_ms']:<10.1f}")

# Overhead results
oh = os.path.join(results_dir, "overhead.json")
if os.path.exists(oh):
    with open(oh) as fh:
        data = json.load(fh)
    print(f"\n=== Overhead ({data.get('gpu_name', '?')}) ===")
    hdr = f"{'Config':<15} {'SeqLen':<8} {'Steps':<6} {'Latency(ms)':<14} {'Overhead%':<10} {'CaptureMs':<10}"
    print(hdr)
    print("-" * len(hdr))
    for r in data["results"]:
        print(f"{r['config']:<15} {r['seq_length']:<8} {r['steps']:<6} {r['avg_latency_ms']:<14.1f} {r['overhead_pct']:<10.1f} {r.get('capture_overhead_ms',0):<10.2f}")

# Compression results
comp = os.path.join(results_dir, "compression.json")
if os.path.exists(comp):
    with open(comp) as fh:
        data = json.load(fh)
    print(f"\n=== Compression ===")
    hdr = f"{'Config':<25} {'SeqLen':<8} {'Steps':<6} {'Ratio':<8} {'ms':<8} {'B/tok':<8}"
    print(hdr)
    print("-" * len(hdr))
    for r in data["results"]:
        print(f"{r['config']:<25} {r['seq_length']:<8} {r['num_steps']:<6} {r['compression_ratio']:<8.1f} {r['compress_time_ms']:<8.1f} {r['bytes_per_token']:<8.1f}")

# Replay results
rep = os.path.join(results_dir, "replay.json")
if os.path.exists(rep):
    with open(rep) as fh:
        data = json.load(fh)
    print(f"\n=== Replay ===")
    hdr = f"{'SeqLen':<8} {'Steps':<6} {'ExactMatch':<12} {'ReplayMs':<10}"
    print(hdr)
    print("-" * len(hdr))
    for r in data["replay_results"]:
        print(f"{r['seq_length']:<8} {r['num_steps']:<6} {r['exact_match_rate']:<12.0%} {r['avg_replay_time_ms']:<10.3f}")
    da = data.get("divergence_analysis", {})
    if da:
        print(f"\nDivergence (no RNG restore): {da.get('divergence_rate', 0):.0%}, avg first step: {da.get('avg_first_divergence_step', -1):.1f}")
