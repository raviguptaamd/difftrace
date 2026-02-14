#!/bin/bash
# Build DiffTrace Docker image on a compute node
# Usage: sbatch slurm/build_docker.slurm  OR  bash docker/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="${DIFFTRACE_IMAGE:-difftrace:rocm6.2}"

echo "=== Building DiffTrace Docker image ==="
echo "Project dir: $PROJECT_DIR"
echo "Image name: $IMAGE_NAME"

cd "$PROJECT_DIR"

docker build \
    -t "$IMAGE_NAME" \
    -f docker/Dockerfile \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "=== Build complete: $IMAGE_NAME ==="
docker images "$IMAGE_NAME"
