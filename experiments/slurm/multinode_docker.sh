#!/bin/bash
# Multi-node DiffTrace benchmark using Docker on each node
# Called by Slurm: srun runs this on every node
#
# Required env vars (set by parent sbatch):
#   MASTER_ADDR, MASTER_PORT, NNODES, DIFFTRACE_DIR
#
# Phase 1: Pull Docker image (all nodes)
# Phase 2: Wait barrier
# Phase 3: Run benchmark

set -euo pipefail

NODE_RANK=${SLURM_NODEID:-0}
NGPUS_PER_NODE=8
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
DIFFTRACE_DIR=${DIFFTRACE_DIR:-$HOME/difftrace}
WORLD_SIZE=$((NNODES * NGPUS_PER_NODE))
PHASE=${PHASE:-run}  # "pull" or "run"

echo "[$(hostname)] Node rank: ${NODE_RANK}/${NNODES}, Master: ${MASTER_ADDR}:${MASTER_PORT}, Phase: ${PHASE}"

if [ "${PHASE}" = "pull" ]; then
    echo "[$(hostname)] Pulling Docker image..."
    docker pull rocm/pytorch:latest 2>&1 | tail -3
    echo "[$(hostname)] Pull done."
    exit 0
fi

# Phase: run
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size 64G \
    --security-opt seccomp=unconfined \
    --net=host \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e MASTER_PORT="${MASTER_PORT}" \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_DEBUG=WARN \
    -e SLURM_NNODES="${NNODES}" \
    -e SLURM_NODEID="${NODE_RANK}" \
    -v "${DIFFTRACE_DIR}":/workspace/difftrace \
    -w /workspace/difftrace \
    rocm/pytorch:latest \
    bash -c "
        pip install -q zstandard 2>&1 | tail -1
        torchrun \
            --nnodes=${NNODES} \
            --nproc_per_node=${NGPUS_PER_NODE} \
            --rdzv_id=\${SLURM_JOB_ID:-${SLURM_JOB_ID:-0}} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            --rdzv_conf=timeout=600 \
            --node_rank=${NODE_RANK} \
            experiments/bench_scalability.py \
                --output-dir results \
                --seq-lengths 256 512 1024 2048 \
                --steps 32 64 128 \
                --num-requests 20 \
                --store-dir /tmp/difftrace_scale_n${NNODES}
    "

echo "[$(hostname)] Done."
