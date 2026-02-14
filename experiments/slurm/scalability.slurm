#!/bin/bash
#SBATCH --job-name=dt-scale
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/scalability_%j.out
#SBATCH --error=logs/scalability_%j.err

# Experiment 4: Multi-GPU Scalability
# Runs with 1, 2, 4, 8 GPUs on a single node

set -euo pipefail

PROJECT_DIR="${DIFFTRACE_DIR:-/home/ravgupta/phantom_amd_llm_operators/difftrace}"
IMAGE_NAME="difftrace:rocm6.2"
RESULTS_DIR="${PROJECT_DIR}/results"

echo "=== DiffTrace Scalability Benchmark ==="
echo "Node: $(hostname)"
echo "GPUs available: $(rocm-smi --showproductname 2>/dev/null | grep -c 'GPU' || echo 'unknown')"
echo "Date: $(date)"

mkdir -p "${RESULTS_DIR}" logs

# Run for different GPU counts
for NGPU in 1 2 4 8; do
    echo ""
    echo "--- Running with ${NGPU} GPUs ---"

    HIP_VISIBLE_DEVICES=""
    for ((i=0; i<NGPU; i++)); do
        if [ -n "$HIP_VISIBLE_DEVICES" ]; then
            HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES},"
        fi
        HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}${i}"
    done

    docker run --rm \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --ipc=host --shm-size 64G \
        -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
        -v "${PROJECT_DIR}:/workspace/difftrace" \
        -v "${RESULTS_DIR}:/workspace/results" \
        -w /workspace/difftrace \
        "$IMAGE_NAME" \
        bash -c "torchrun --nproc_per_node=${NGPU} experiments/bench_scalability.py \
            --output-dir /workspace/results \
            --seq-lengths 256 512 1024 \
            --steps 32 64 \
            --num-requests 20 \
            --store-dir /tmp/difftrace_scale_${NGPU}"

    echo "--- ${NGPU} GPUs complete ---"
done

echo ""
echo "=== All scalability benchmarks complete ==="
echo "Results: ${RESULTS_DIR}/scalability_w*.json"
