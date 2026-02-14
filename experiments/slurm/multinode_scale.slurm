#!/bin/bash
#SBATCH --job-name=dt-mnode
#SBATCH --partition=amd-rccl
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/scale_n%N_j%j.out
#SBATCH --error=logs/scale_n%N_j%j.err

# Multi-node DiffTrace Scalability Benchmark
# Submit with: sbatch --nodes=N experiments/slurm/multinode_scale.slurm
# where N = 3, 5, 7, or 9

set -euo pipefail

PROJECT_DIR="${DIFFTRACE_DIR:-$(pwd)}"
RESULTS_DIR="${PROJECT_DIR}/results"
IMAGE_NAME="${DIFFTRACE_IMAGE:-difftrace:cluster}"
NNODES=${SLURM_NNODES}
NGPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * NGPUS_PER_NODE))

# Pick master from first node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "============================================"
echo "DiffTrace Multi-Node Scalability Benchmark"
echo "============================================"
echo "Nodes: ${NNODES}"
echo "GPUs/node: ${NGPUS_PER_NODE}"
echo "Total GPUs: ${WORLD_SIZE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "Date: $(date)"
echo "============================================"

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"

# Use srun to launch across all nodes
# Each node runs 8 GPU processes via torchrun
srun --nodes=${NNODES} --ntasks-per-node=1 \
    bash -c "
        export MASTER_ADDR=${MASTER_ADDR}
        export MASTER_PORT=${MASTER_PORT}
        export NCCL_SOCKET_IFNAME=eth0
        export NCCL_DEBUG=WARN

        echo \"[\$(hostname)] Starting torchrun with ${NGPUS_PER_NODE} GPUs\"

        cd ${PROJECT_DIR}
        torchrun \
            --nnodes=${NNODES} \
            --nproc_per_node=${NGPUS_PER_NODE} \
            --rdzv_id=\${SLURM_JOB_ID} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            --node_rank=\${SLURM_NODEID} \
            experiments/bench_scalability.py \
                --output-dir ${RESULTS_DIR} \
                --seq-lengths 256 512 1024 2048 \
                --steps 32 64 128 \
                --num-requests 20 \
                --store-dir /tmp/difftrace_scale_n${NNODES}
    "

echo ""
echo "=== Completed: ${NNODES} nodes, ${WORLD_SIZE} GPUs ==="
echo "Results: ${RESULTS_DIR}/scalability_w${WORLD_SIZE}.json"
