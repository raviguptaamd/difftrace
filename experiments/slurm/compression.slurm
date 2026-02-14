#!/bin/bash
#SBATCH --job-name=dt-compress
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/compression_%j.out
#SBATCH --error=logs/compression_%j.err

# Experiment 2: Differential Compression Evaluation
# CPU-only benchmark (no GPU needed for compression-only testing)

set -euo pipefail

PROJECT_DIR="${DIFFTRACE_DIR:-/home/ravgupta/phantom_amd_llm_operators/difftrace}"
IMAGE_NAME="difftrace:rocm6.2"
RESULTS_DIR="${PROJECT_DIR}/results"

echo "=== DiffTrace Compression Benchmark ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

mkdir -p "${RESULTS_DIR}" logs

docker run --rm \
    --ipc=host --shm-size 32G \
    -v "${PROJECT_DIR}:/workspace/difftrace" \
    -v "${RESULTS_DIR}:/workspace/results" \
    -w /workspace/difftrace \
    "$IMAGE_NAME" \
    python experiments/bench_compression.py \
        --output /workspace/results/compression.json \
        --seq-lengths 128 256 512 1024 2048 4096 \
        --steps 16 32 64 128 256 \
        --vocab-size 32000 \
        --error-bounds 0.1 0.01 0.001 0.0001

echo "=== Compression benchmark complete ==="
echo "Results: ${RESULTS_DIR}/compression.json"
