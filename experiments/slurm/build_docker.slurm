#!/bin/bash
#SBATCH --job-name=difftrace-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

# Build DiffTrace Docker image on a compute node
# (Docker builds should not run on login nodes)

set -euo pipefail

PROJECT_DIR="${DIFFTRACE_DIR:-/home/ravgupta/phantom_amd_llm_operators/difftrace}"
IMAGE_NAME="difftrace:rocm6.2"

echo "=== Building DiffTrace Docker image ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Project: $PROJECT_DIR"

cd "$PROJECT_DIR"

mkdir -p logs

# Build the Docker image
docker build \
    -t "$IMAGE_NAME" \
    -f docker/Dockerfile \
    .

echo "=== Build complete ==="
docker images "$IMAGE_NAME"
