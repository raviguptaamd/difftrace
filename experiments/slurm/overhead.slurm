#!/bin/bash
#SBATCH --job-name=dt-overhead
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/overhead_%j.out
#SBATCH --error=logs/overhead_%j.err

# Experiment 1: DiffTrace Overhead Measurement
# Single GPU benchmark comparing baseline vs DiffTrace at different granularities

set -euo pipefail

PROJECT_DIR="${DIFFTRACE_DIR:-/home/ravgupta/phantom_amd_llm_operators/difftrace}"
IMAGE_NAME="difftrace:rocm6.2"
RESULTS_DIR="${PROJECT_DIR}/results"
MODEL_DIR="${MODEL_DIR:-/models}"

echo "=== DiffTrace Overhead Benchmark ==="
echo "Node: $(hostname)"
echo "GPU: $(rocm-smi --showproductname 2>/dev/null | head -5 || echo 'unknown')"
echo "Date: $(date)"

mkdir -p "${RESULTS_DIR}" logs

# Run overhead benchmark
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size 64G \
    -v "${PROJECT_DIR}:/workspace/difftrace" \
    -v "${RESULTS_DIR}:/workspace/results" \
    -v "${MODEL_DIR}:/models" \
    -w /workspace/difftrace \
    "$IMAGE_NAME" \
    python experiments/bench_overhead.py \
        --device cuda:0 \
        --output /workspace/results/overhead.json \
        --seq-lengths 128 256 512 1024 2048 \
        --steps 16 32 64 128 \
        --num-warmup 3 \
        --num-runs 10 \
        --vocab-size 32000 \
        --hidden-dim 4096

echo "=== Overhead benchmark complete ==="
echo "Results: ${RESULTS_DIR}/overhead.json"
