#!/bin/bash
# =============================================================================
# DiffTrace: Full experiment pipeline for AMD MI300X cluster
# (useocpslog-002.amd.com)
#
# Usage:
#   1. Copy difftrace/ directory to cluster:
#      rsync -avz difftrace/ ravgupta@useocpslog-002.amd.com:~/difftrace/
#
#   2. SSH to cluster and run:
#      cd ~/difftrace && bash experiments/slurm/run_cluster.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${PROJECT_DIR}/logs"

echo "=== DiffTrace Cluster Pipeline ==="
echo "Project: ${PROJECT_DIR}"
echo "Cluster: $(hostname)"
echo "Date: $(date)"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# ── Step 1: Build Docker image ──────────────────────────────────────────────
echo ""
echo "Step 1: Building Docker image on compute node..."
BUILD_JOB=$(sbatch --parsable \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=00:30:00 \
    --output="${LOGS_DIR}/build_%j.out" \
    --error="${LOGS_DIR}/build_%j.err" \
    --wrap="cd ${PROJECT_DIR} && docker build -t difftrace:cluster -f docker/Dockerfile.cluster .")
echo "  Build job: ${BUILD_JOB}"

# ── Step 2: CPU-only benchmarks (compression + replay) ──────────────────────
echo ""
echo "Step 2: Submitting CPU benchmarks (compression + replay)..."
CPU_JOB=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --time=01:00:00 \
    --output="${LOGS_DIR}/cpu_bench_%j.out" \
    --error="${LOGS_DIR}/cpu_bench_%j.err" \
    --wrap="cd ${PROJECT_DIR} && \
docker run --rm --ipc=host --shm-size 32G \
    -v ${PROJECT_DIR}:/workspace/difftrace \
    -w /workspace/difftrace \
    difftrace:cluster \
    bash -c '
        python experiments/bench_compression.py \
            --output results/compression.json \
            --seq-lengths 128 256 512 1024 2048 \
            --steps 16 32 64 128 \
            --vocab-size 4000 \
            --error-bounds 0.1 0.01 0.001 0.0001 && \
        python experiments/bench_replay.py \
            --output results/replay.json \
            --seq-lengths 128 256 512 1024 2048 \
            --steps 16 32 64 128 \
            --num-trials 20
    '")
echo "  CPU benchmarks job: ${CPU_JOB}"

# ── Step 3: Single-GPU overhead benchmark ────────────────────────────────────
echo ""
echo "Step 3: Submitting overhead benchmark (1 GPU)..."
OH_JOB=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gpus=1 \
    --time=02:00:00 \
    --output="${LOGS_DIR}/overhead_%j.out" \
    --error="${LOGS_DIR}/overhead_%j.err" \
    --wrap="cd ${PROJECT_DIR} && \
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size 64G \
    -v ${PROJECT_DIR}:/workspace/difftrace \
    -w /workspace/difftrace \
    difftrace:cluster \
    python experiments/bench_overhead.py \
        --device cuda:0 \
        --output results/overhead.json \
        --seq-lengths 128 256 512 1024 2048 \
        --steps 16 32 64 128 \
        --num-warmup 3 \
        --num-runs 10")
echo "  Overhead job: ${OH_JOB}"

# ── Step 4: Multi-GPU scalability ────────────────────────────────────────────
echo ""
echo "Step 4: Submitting scalability benchmarks (1,2,4,8 GPUs)..."
SCALE_JOB=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gpus=8 \
    --time=02:00:00 \
    --output="${LOGS_DIR}/scalability_%j.out" \
    --error="${LOGS_DIR}/scalability_%j.err" \
    --wrap="cd ${PROJECT_DIR} && \
for NGPU in 1 2 4 8; do
    echo \"--- Running with \${NGPU} GPUs ---\"
    HIP_DEVS=\$(seq -s, 0 \$((NGPU-1)))
    docker run --rm \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --ipc=host --shm-size 64G \
        -e HIP_VISIBLE_DEVICES=\${HIP_DEVS} \
        -v ${PROJECT_DIR}:/workspace/difftrace \
        -w /workspace/difftrace \
        difftrace:cluster \
        bash -c \"torchrun --nproc_per_node=\${NGPU} experiments/bench_scalability.py \
            --output-dir results \
            --seq-lengths 256 512 1024 \
            --steps 32 64 \
            --num-requests 20\"
done")
echo "  Scalability job: ${SCALE_JOB}"

# ── Step 5: Generate plots (after all experiments) ───────────────────────────
echo ""
echo "Step 5: Submitting plot generation..."
PLOT_JOB=$(sbatch --parsable \
    --dependency=afterok:${CPU_JOB}:${OH_JOB}:${SCALE_JOB} \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --time=00:10:00 \
    --output="${LOGS_DIR}/plots_%j.out" \
    --error="${LOGS_DIR}/plots_%j.err" \
    --wrap="cd ${PROJECT_DIR} && \
docker run --rm \
    -v ${PROJECT_DIR}:/workspace/difftrace \
    -w /workspace/difftrace \
    difftrace:cluster \
    python analysis/plot_all.py \
        --results-dir results \
        --output-dir paper/figures")
echo "  Plot generation job: ${PLOT_JOB}"

echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Job chain:"
echo "  Build:       ${BUILD_JOB}"
echo "  CPU Bench:   ${CPU_JOB} (depends on build)"
echo "  Overhead:    ${OH_JOB} (depends on build)"
echo "  Scalability: ${SCALE_JOB} (depends on build)"
echo "  Plots:       ${PLOT_JOB} (depends on all benchmarks)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${RESULTS_DIR}/"
echo "Figures: ${PROJECT_DIR}/paper/figures/"
