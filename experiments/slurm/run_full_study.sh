#!/bin/bash
# =============================================================================
# DiffTrace: Full Multi-Node Study on AMD MI300X Cluster
#
# Runs experiments at 3, 5, 7, 9 nodes (24, 40, 56, 72 GPUs)
# plus single-node baselines (overhead, compression, replay)
#
# Usage:
#   cd ~/difftrace && bash experiments/slurm/run_full_study.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${PROJECT_DIR}/logs"

export DIFFTRACE_DIR="$PROJECT_DIR"
export DIFFTRACE_IMAGE="difftrace:cluster"

echo "=========================================================="
echo "  DiffTrace: Full Multi-Node Experiment Pipeline"
echo "=========================================================="
echo "Project:  ${PROJECT_DIR}"
echo "Cluster:  $(hostname)"
echo "Date:     $(date)"
echo ""

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# ── Step 0: Install DiffTrace locally (no Docker needed for pure Python) ─────
echo "Step 0: Installing DiffTrace and dependencies..."
cd "${PROJECT_DIR}"
pip install --user -q zstandard numpy torch 2>/dev/null || true
pip install --user -e . 2>/dev/null || true
echo "  Done."

# ── Step 1: Single-node CPU benchmarks (compression + replay) ────────────────
echo ""
echo "Step 1: Submitting CPU benchmarks (compression + replay)..."
CPU_JOB=$(sbatch --parsable \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --time=00:30:00 \
    --output="${LOGS_DIR}/cpu_bench_%j.out" \
    --error="${LOGS_DIR}/cpu_bench_%j.err" \
    --wrap="
        cd ${PROJECT_DIR}
        echo '=== Compression Benchmark ===' 
        python experiments/bench_compression.py \
            --output results/compression.json \
            --seq-lengths 128 256 512 1024 2048 \
            --steps 16 32 64 128 \
            --vocab-size 4000 \
            --error-bounds 0.1 0.01 0.001 0.0001
        echo '=== Replay Benchmark ==='
        python experiments/bench_replay.py \
            --output results/replay.json \
            --seq-lengths 128 256 512 1024 2048 \
            --steps 16 32 64 128 \
            --num-trials 20
    ")
echo "  CPU benchmarks job: ${CPU_JOB}"

# ── Step 2: Single-node GPU overhead benchmark ───────────────────────────────
echo ""
echo "Step 2: Submitting single-node overhead benchmark (8 GPUs)..."
OH_JOB=$(sbatch --parsable \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gpus-per-node=8 \
    --exclusive \
    --time=01:00:00 \
    --output="${LOGS_DIR}/overhead_%j.out" \
    --error="${LOGS_DIR}/overhead_%j.err" \
    --wrap="
        cd ${PROJECT_DIR}
        echo '=== Single-GPU Overhead ==='
        python experiments/bench_overhead.py \
            --device cuda:0 \
            --output results/overhead.json \
            --seq-lengths 128 256 512 1024 2048 \
            --steps 16 32 64 128 \
            --num-warmup 3 \
            --num-runs 10

        echo '=== Single-Node Scalability (1,2,4,8 GPUs) ==='
        for NGPU in 1 2 4 8; do
            echo \"--- \${NGPU} GPUs ---\"
            torchrun --nproc_per_node=\${NGPU} --master_port=29501 \
                experiments/bench_scalability.py \
                    --output-dir results \
                    --seq-lengths 256 512 1024 2048 \
                    --steps 32 64 128 \
                    --num-requests 20 \
                    --store-dir /tmp/difftrace_scale_\${NGPU}
        done
    ")
echo "  Overhead + single-node scale job: ${OH_JOB}"

# ── Step 3: Multi-node scalability at 3, 5, 7, 9 nodes ──────────────────────
echo ""
echo "Step 3: Submitting multi-node scalability benchmarks..."
SCALE_JOBS=""
for NNODES in 3 5 7 9; do
    NGPUS=$((NNODES * 8))
    echo "  Submitting ${NNODES} nodes (${NGPUS} GPUs)..."
    JOB=$(sbatch --parsable \
        --partition=amd-rccl \
        --nodes=${NNODES} \
        --ntasks-per-node=1 \
        --cpus-per-task=64 \
        --gpus-per-node=8 \
        --exclusive \
        --time=01:00:00 \
        --output="${LOGS_DIR}/scale_n${NNODES}_%j.out" \
        --error="${LOGS_DIR}/scale_n${NNODES}_%j.err" \
        --wrap="
            cd ${PROJECT_DIR}
            MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)
            MASTER_PORT=\$((29500 + ${NNODES}))
            export NCCL_SOCKET_IFNAME=eth0
            export NCCL_DEBUG=WARN
            
            echo '=== Multi-Node Scalability: ${NNODES} nodes, ${NGPUS} GPUs ==='
            echo \"Master: \${MASTER_ADDR}:\${MASTER_PORT}\"
            echo \"Nodes: \${SLURM_JOB_NODELIST}\"
            
            srun --nodes=${NNODES} --ntasks-per-node=1 \
                bash -c \"
                    torchrun \
                        --nnodes=${NNODES} \
                        --nproc_per_node=8 \
                        --rdzv_id=\\\${SLURM_JOB_ID} \
                        --rdzv_backend=c10d \
                        --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
                        --node_rank=\\\${SLURM_NODEID} \
                        ${PROJECT_DIR}/experiments/bench_scalability.py \
                            --output-dir ${RESULTS_DIR} \
                            --seq-lengths 256 512 1024 2048 \
                            --steps 32 64 128 \
                            --num-requests 20 \
                            --store-dir /tmp/difftrace_scale_n${NNODES}
                \"
        ")
    echo "    Job: ${JOB}"
    SCALE_JOBS="${SCALE_JOBS}:${JOB}"
done

# ── Step 4: Generate all plots after experiments ─────────────────────────────
echo ""
echo "Step 4: Submitting plot generation (after all experiments)..."

# Build dependency string (all previous jobs)
ALL_DEPS="${CPU_JOB}:${OH_JOB}${SCALE_JOBS}"
PLOT_JOB=$(sbatch --parsable \
    --dependency=afterany:${ALL_DEPS} \
    --partition=amd-rccl \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --time=00:10:00 \
    --output="${LOGS_DIR}/plots_%j.out" \
    --error="${LOGS_DIR}/plots_%j.err" \
    --wrap="
        cd ${PROJECT_DIR}
        python analysis/plot_all.py \
            --results-dir results \
            --output-dir paper/figures
        echo '=== All plots generated ==='
        ls -la paper/figures/
    ")
echo "  Plot job: ${PLOT_JOB}"

echo ""
echo "=========================================================="
echo "  All jobs submitted!"
echo "=========================================================="
echo ""
echo "Job Summary:"
echo "  CPU Bench (compress+replay):      ${CPU_JOB}"
echo "  Overhead + 1-node scale:          ${OH_JOB}"
for NNODES in 3 5 7 9; do
    echo "  Multi-node scale (${NNODES} nodes):     (see above)"
done
echo "  Plot generation:                  ${PLOT_JOB}"
echo ""
echo "Monitor:    squeue -u \$USER"
echo "Results:    ${RESULTS_DIR}/"
echo "Figures:    ${PROJECT_DIR}/paper/figures/"
echo "Logs:       ${LOGS_DIR}/"
echo ""
echo "After completion, sync results back:"
echo "  rsync -avz ${RESULTS_DIR}/ local:results/"
echo "  rsync -avz ${PROJECT_DIR}/paper/figures/ local:paper/figures/"
