#!/usr/bin/env python3
"""
Generate all plots for the DiffTrace paper.

Reads experiment results from results/ and generates publication-quality
figures for the IEEE paper in paper/figures/.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Paper-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# IEEE column width: 3.5 inches (single), 7.16 inches (double)
SINGLE_COL = 3.5
DOUBLE_COL = 7.16
COLORS = sns.color_palette("Set2", 8)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plot_all")


def plot_overhead(results_path: str, output_dir: str):
    """
    Figure 2: DiffTrace overhead at different capture granularities.
    Two panels: (a) Absolute latency, (b) Overhead percentage.
    """
    with open(results_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["results"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # (a) Absolute latency vs sequence length (fixed steps=64)
    df_64 = df[df["steps"] == 64]
    configs = ["baseline", "minimal", "medium", "full_async", "full_sync"]
    labels = ["Baseline", "Tokens Only", "Tokens+Masks", "Full (Async)", "Full (Sync)"]
    markers = ["o", "s", "D", "^", "v"]

    for i, (cfg, label) in enumerate(zip(configs, labels)):
        subset = df_64[df_64["config"] == cfg].sort_values("seq_length")
        if len(subset) > 0:
            ax1.plot(
                subset["seq_length"], subset["avg_latency_ms"],
                marker=markers[i], color=COLORS[i], label=label,
                linewidth=1.5, markersize=4,
            )

    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("(a) Inference Latency")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.grid(True, alpha=0.3)

    # (b) Overhead % vs sequence length
    df_overhead = df_64[df_64["config"] != "baseline"]
    for i, (cfg, label) in enumerate(zip(configs[1:], labels[1:])):
        subset = df_overhead[df_overhead["config"] == cfg].sort_values("seq_length")
        if len(subset) > 0:
            ax2.plot(
                subset["seq_length"], subset["overhead_pct"],
                marker=markers[i+1], color=COLORS[i+1], label=label,
                linewidth=1.5, markersize=4,
            )

    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Overhead (%)")
    ax2.set_title("(b) DiffTrace Overhead")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.axhline(y=5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_overhead.pdf"))
    plt.savefig(os.path.join(output_dir, "fig_overhead.png"))
    plt.close()
    logger.info("Generated: fig_overhead.pdf")


def plot_compression(results_path: str, output_dir: str):
    """
    Figure 3: Compression effectiveness.
    (a) Compression ratio vs sequence length
    (b) Storage per token vs error bound (lossy)
    """
    with open(results_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["results"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # (a) Compression ratio vs seq length (steps=64)
    df_64 = df[df["num_steps"] == 64]
    lossless_configs = ["none", "diff_lossless", "diff_lossless_high"]
    lossless_labels = ["No Compression", "Diff + zstd (L3)", "Diff + zstd (L9)"]

    for i, (cfg, label) in enumerate(zip(lossless_configs, lossless_labels)):
        subset = df_64[df_64["config"] == cfg].sort_values("seq_length")
        if len(subset) > 0:
            ax1.plot(
                subset["seq_length"], subset["compression_ratio"],
                marker="osDv"[i], color=COLORS[i], label=label,
                linewidth=1.5, markersize=4,
            )

    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Compression Ratio")
    ax1.set_title("(a) Lossless Compression")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.grid(True, alpha=0.3)

    # (b) Storage per token for lossy configs at different error bounds
    lossy_df = df_64[df_64["config"].str.startswith("diff_lossy")]
    if len(lossy_df) > 0:
        # Extract error bounds from config names
        lossy_df = lossy_df.copy()
        lossy_df["error_bound"] = lossy_df["config"].apply(
            lambda x: float(x.split("eb")[-1]) if "eb" in x else 0
        )

        for seq_len in [256, 512, 1024]:
            subset = lossy_df[lossy_df["seq_length"] == seq_len].sort_values("error_bound")
            if len(subset) > 0:
                ax2.plot(
                    subset["error_bound"], subset["bytes_per_token"],
                    marker="o", label=f"L={seq_len}",
                    linewidth=1.5, markersize=4,
                )

    ax2.set_xlabel("Error Bound (ε)")
    ax2.set_ylabel("Bytes / Token")
    ax2.set_title("(b) Lossy: Storage vs Accuracy")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_compression.pdf"))
    plt.savefig(os.path.join(output_dir, "fig_compression.png"))
    plt.close()
    logger.info("Generated: fig_compression.pdf")


def plot_replay(results_path: str, output_dir: str):
    """
    Figure 4: Replay accuracy and performance.
    (a) Token match rate heatmap (seq_length x steps)
    (b) Replay speedup bar chart
    """
    with open(results_path) as f:
        data = json.load(f)

    replay_df = pd.DataFrame(data["replay_results"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # (a) Exact match rate heatmap
    pivot = replay_df.pivot_table(
        values="exact_match_rate",
        index="seq_length",
        columns="num_steps",
        aggfunc="mean",
    )
    if len(pivot) > 0:
        sns.heatmap(
            pivot, annot=True, fmt=".0%", cmap="YlGn",
            vmin=0, vmax=1, ax=ax1,
            cbar_kws={"label": "Exact Match Rate"},
        )
    ax1.set_title("(a) Exact Match Rate")
    ax1.set_xlabel("Denoising Steps")
    ax1.set_ylabel("Sequence Length")

    # (b) Replay speedup
    if "avg_speedup" in replay_df.columns:
        replay_64 = replay_df[replay_df["num_steps"] == 64].sort_values("seq_length")
        if len(replay_64) > 0:
            bars = ax2.bar(
                range(len(replay_64)),
                replay_64["avg_speedup"],
                color=COLORS[0],
                edgecolor="black",
                linewidth=0.5,
            )
            ax2.set_xticks(range(len(replay_64)))
            ax2.set_xticklabels(replay_64["seq_length"].values)
            ax2.set_xlabel("Sequence Length")
            ax2.set_ylabel("Speedup (Token Replay / Original)")
            ax2.set_title("(b) Token Replay Speedup")
            ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
            ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_replay.pdf"))
    plt.savefig(os.path.join(output_dir, "fig_replay.png"))
    plt.close()
    logger.info("Generated: fig_replay.pdf")


def plot_scalability(results_dir: str, output_dir: str):
    """
    Figure 5: Multi-node multi-GPU scalability on AMD MI300X.
    (a) Overhead % vs total GPUs (1-72)
    (b) Compression ratio vs total GPUs
    """
    # Load results for ALL world sizes (single-node + multi-node)
    all_data = []
    world_sizes_found = []
    for w in [1, 2, 4, 8, 24, 40, 56, 72]:
        path = os.path.join(results_dir, f"scalability_w{w}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                num_nodes = data.get("num_nodes", max(1, w // 8))
                for r in data["results"]:
                    r["num_nodes"] = num_nodes
                    r["gpus_per_node"] = data.get("gpus_per_node", min(w, 8))
                all_data.extend(data["results"])
                world_sizes_found.append(w)

    if not all_data:
        logger.warning("No scalability results found")
        return

    df = pd.DataFrame(all_data)
    df_r0 = df[df["rank"] == 0]

    logger.info(f"Scalability: found world sizes {world_sizes_found}")

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5))
    ax1, ax2, ax3 = axes

    # Separate single-node and multi-node
    single_ws = sorted([w for w in world_sizes_found if w <= 8])
    multi_ws = sorted([w for w in world_sizes_found if w > 8])
    all_ws = sorted(world_sizes_found)

    # Node labels for x-axis
    def gpu_label(w):
        if w <= 8:
            return f"{w}G\n1N"
        return f"{w}G\n{w//8}N"

    # (a) Overhead % vs total GPUs for steps=64
    seq_colors = {256: COLORS[0], 512: COLORS[1], 1024: COLORS[2], 2048: COLORS[3]}
    seq_markers = {256: "o", 512: "s", 1024: "D", 2048: "^"}

    for seq_len in sorted(df_r0["seq_length"].unique()):
        subset = df_r0[
            (df_r0["seq_length"] == seq_len) & (df_r0["num_steps"] == 64)
        ].sort_values("world_size")
        if len(subset) > 0:
            ax1.plot(
                range(len(subset)), subset["overhead_pct"].values,
                marker=seq_markers.get(seq_len, "o"),
                color=seq_colors.get(seq_len, COLORS[0]),
                label=f"L={seq_len}",
                linewidth=1.5, markersize=4,
            )
            ws_vals = subset["world_size"].values
            ax1.set_xticks(range(len(ws_vals)))
            ax1.set_xticklabels([gpu_label(w) for w in ws_vals], fontsize=6)

    ax1.set_xlabel("GPUs / Nodes")
    ax1.set_ylabel("Overhead (%)")
    ax1.set_title("(a) Overhead vs Scale")
    ax1.legend(fontsize=6, framealpha=0.9, loc="upper left")
    ax1.axhline(y=5, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="5% target")
    ax1.grid(True, alpha=0.3)

    # (b) Compression ratio vs total GPUs (steps=64)
    for seq_len in sorted(df_r0["seq_length"].unique()):
        subset = df_r0[
            (df_r0["seq_length"] == seq_len) & (df_r0["num_steps"] == 64)
        ].sort_values("world_size")
        if len(subset) > 0:
            ax2.plot(
                range(len(subset)), subset["compression_ratio"].values,
                marker=seq_markers.get(seq_len, "s"),
                color=seq_colors.get(seq_len, COLORS[0]),
                label=f"L={seq_len}",
                linewidth=1.5, markersize=4,
            )
            ws_vals = subset["world_size"].values
            ax2.set_xticks(range(len(ws_vals)))
            ax2.set_xticklabels([gpu_label(w) for w in ws_vals], fontsize=6)

    ax2.set_xlabel("GPUs / Nodes")
    ax2.set_ylabel("Compression Ratio")
    ax2.set_title("(b) Compression at Scale")
    ax2.legend(fontsize=6, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # (c) Per-request time vs total GPUs (steps=64, seq=1024)
    for num_steps in sorted(df_r0["num_steps"].unique()):
        subset = df_r0[
            (df_r0["seq_length"] == 1024) & (df_r0["num_steps"] == num_steps)
        ].sort_values("world_size")
        if len(subset) > 0:
            ax3.plot(
                range(len(subset)), subset["time_per_request_ms"].values,
                marker="o", label=f"T={num_steps}",
                linewidth=1.5, markersize=4,
            )
            ws_vals = subset["world_size"].values
            ax3.set_xticks(range(len(ws_vals)))
            ax3.set_xticklabels([gpu_label(w) for w in ws_vals], fontsize=6)

    ax3.set_xlabel("GPUs / Nodes")
    ax3.set_ylabel("Provenance Time (ms/req)")
    ax3.set_title("(c) L=1024: Time/Request")
    ax3.legend(fontsize=6, framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_scalability.pdf"))
    plt.savefig(os.path.join(output_dir, "fig_scalability.png"))
    plt.close()
    logger.info("Generated: fig_scalability.pdf")

    # Also generate a focused multi-node figure
    if multi_ws:
        _plot_multinode_scaling(df_r0, multi_ws, seq_colors, seq_markers, output_dir)


def _plot_multinode_scaling(df_r0, multi_ws, seq_colors, seq_markers, output_dir):
    """
    Figure 6: Multi-node scaling (3, 5, 7, 9 nodes).
    Focused view on cross-node overhead behavior.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    df_multi = df_r0[df_r0["world_size"].isin(multi_ws)]
    node_counts = sorted(df_multi["world_size"].unique())
    node_labels = [f"{w//8}N ({w}G)" for w in node_counts]

    # (a) Overhead grouped bar chart by seq_length at steps=128 (most favorable)
    steps_val = 128
    bar_width = 0.2
    seq_lens = sorted(df_multi["seq_length"].unique())
    x = np.arange(len(node_counts))

    for i, seq_len in enumerate(seq_lens):
        subset = df_multi[
            (df_multi["seq_length"] == seq_len) & (df_multi["num_steps"] == steps_val)
        ].sort_values("world_size")
        if len(subset) > 0:
            vals = [subset[subset["world_size"] == w]["overhead_pct"].values[0]
                    if w in subset["world_size"].values else 0 for w in node_counts]
            ax1.bar(
                x + i * bar_width - bar_width * len(seq_lens) / 2,
                vals, bar_width,
                label=f"L={seq_len}",
                color=seq_colors.get(seq_len, COLORS[i]),
                edgecolor="black", linewidth=0.3,
            )

    ax1.set_xlabel("Nodes (GPUs)")
    ax1.set_ylabel("Overhead (%)")
    ax1.set_title(f"(a) Multi-Node Overhead (T={steps_val})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(node_labels, fontsize=7)
    ax1.legend(fontsize=6, framealpha=0.9)
    ax1.axhline(y=10, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis="y")

    # (b) Per-request provenance time vs nodes for different step counts (seq=1024)
    for num_steps in sorted(df_multi["num_steps"].unique()):
        subset = df_multi[
            (df_multi["seq_length"] == 1024) & (df_multi["num_steps"] == num_steps)
        ].sort_values("world_size")
        if len(subset) > 0:
            ax2.plot(
                range(len(subset)), subset["time_per_request_ms"].values,
                marker="o", label=f"T={num_steps}",
                linewidth=1.5, markersize=4,
            )

    ax2.set_xticks(range(len(node_counts)))
    ax2.set_xticklabels(node_labels, fontsize=7)
    ax2.set_xlabel("Nodes (GPUs)")
    ax2.set_ylabel("Provenance Time (ms/req)")
    ax2.set_title("(b) L=1024: Provenance Pipeline")
    ax2.legend(fontsize=6, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_multinode.pdf"))
    plt.savefig(os.path.join(output_dir, "fig_multinode.png"))
    plt.close()
    logger.info("Generated: fig_multinode.pdf")


def plot_architecture(output_dir: str):
    """
    Figure 1: DiffTrace system architecture diagram.
    Created programmatically for consistent style.
    """
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Component boxes
    boxes = [
        (0.5, 5.5, 3, 1.2, "dLLM\nDenoising Loop", COLORS[0]),
        (4, 5.5, 2.5, 1.2, "Trajectory\nCapture", COLORS[1]),
        (7, 5.5, 2.5, 1.2, "RNG State\nCapture", COLORS[2]),
        (4, 3.5, 2.5, 1.2, "Differential\nCompressor", COLORS[3]),
        (7, 3.5, 2.5, 1.2, "Async I/O\nEngine", COLORS[4]),
        (0.5, 1.5, 3, 1.2, "Provenance\nStore", COLORS[5]),
        (4, 1.5, 2.5, 1.2, "Replay\nEngine", COLORS[6]),
        (7, 1.5, 2.5, 1.2, "Distributed\nCoordinator", COLORS[7]),
    ]

    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor="black",
                             facecolor=color, alpha=0.7, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
               fontsize=7, fontweight="bold", zorder=3)

    # Arrows
    arrows = [
        (3.5, 6.1, 0.5, 0, "step\ndata"),
        (6.5, 6.1, 0.5, 0, "RNG"),
        (5.25, 5.5, 0, -0.8, ""),
        (8.25, 5.5, 0, -0.8, ""),
        (5.25, 3.5, -2.25, -0.8, ""),
        (8.25, 3.5, 0, -0.8, ""),
        (3.5, 2.1, 0.5, 0, "query"),
        (8.25, 2.7, 0, 0.8, ""),
    ]

    for x, y, dx, dy, label in arrows:
        ax.annotate(
            "", xy=(x + dx, y + dy), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            zorder=1,
        )
        if label:
            ax.text(x + dx/2, y + dy/2 + 0.15, label,
                   ha="center", va="bottom", fontsize=5, style="italic")

    ax.set_title("DiffTrace System Architecture", fontsize=10, fontweight="bold", pad=5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_architecture.pdf"))
    plt.savefig(os.path.join(output_dir, "fig_architecture.png"))
    plt.close()
    logger.info("Generated: fig_architecture.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate DiffTrace paper figures")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="paper/figures", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Always generate architecture diagram
    plot_architecture(args.output_dir)

    # Generate data-dependent plots if results exist
    overhead_path = os.path.join(args.results_dir, "overhead.json")
    if os.path.exists(overhead_path):
        plot_overhead(overhead_path, args.output_dir)
    else:
        logger.warning(f"No overhead results at {overhead_path}")

    compression_path = os.path.join(args.results_dir, "compression.json")
    if os.path.exists(compression_path):
        plot_compression(compression_path, args.output_dir)
    else:
        logger.warning(f"No compression results at {compression_path}")

    replay_path = os.path.join(args.results_dir, "replay.json")
    if os.path.exists(replay_path):
        plot_replay(replay_path, args.output_dir)
    else:
        logger.warning(f"No replay results at {replay_path}")

    plot_scalability(args.results_dir, args.output_dir)

    logger.info(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
