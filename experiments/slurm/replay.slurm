#!/bin/bash
#SBATCH --job-name=dt-replay
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/replay_%j.out
#SBATCH --error=logs/replay_%j.err

# Experiment 3: Replay Accuracy and Performance

set -euo pipefail

PROJECT_DIR="${DIFFTRACE_DIR:-/home/ravgupta/phantom_amd_llm_operators/difftrace}"
IMAGE_NAME="difftrace:rocm6.2"
RESULTS_DIR="${PROJECT_DIR}/results"

echo "=== DiffTrace Replay Benchmark ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

mkdir -p "${RESULTS_DIR}" logs

docker run --rm \
    --ipc=host --shm-size 32G \
    -v "${PROJECT_DIR}:/workspace/difftrace" \
    -v "${RESULTS_DIR}:/workspace/results" \
    -w /workspace/difftrace \
    "$IMAGE_NAME" \
    python experiments/bench_replay.py \
        --output /workspace/results/replay.json \
        --seq-lengths 128 256 512 1024 2048 \
        --steps 16 32 64 128 \
        --num-trials 20 \
        --vocab-size 32000

echo "=== Replay benchmark complete ==="
echo "Results: ${RESULTS_DIR}/replay.json"
