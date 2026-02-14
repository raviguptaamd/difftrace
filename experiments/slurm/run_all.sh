#!/bin/bash
# Submit all DiffTrace experiments to Slurm
# Usage: bash experiments/slurm/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export DIFFTRACE_DIR="$PROJECT_DIR"

echo "=== Submitting DiffTrace Experiments ==="
echo "Project: $PROJECT_DIR"
echo ""

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results"

# 1. Build Docker image first
echo "Submitting Docker build..."
BUILD_JOB=$(sbatch --parsable "$SCRIPT_DIR/build_docker.slurm")
echo "  Build job: $BUILD_JOB"

# 2. Submit experiments with dependency on build
echo "Submitting overhead benchmark..."
OH_JOB=$(sbatch --parsable --dependency=afterok:${BUILD_JOB} "$SCRIPT_DIR/overhead.slurm")
echo "  Overhead job: $OH_JOB"

echo "Submitting compression benchmark..."
COMP_JOB=$(sbatch --parsable --dependency=afterok:${BUILD_JOB} "$SCRIPT_DIR/compression.slurm")
echo "  Compression job: $COMP_JOB"

echo "Submitting replay benchmark..."
REP_JOB=$(sbatch --parsable --dependency=afterok:${BUILD_JOB} "$SCRIPT_DIR/replay.slurm")
echo "  Replay job: $REP_JOB"

echo "Submitting scalability benchmark..."
SCALE_JOB=$(sbatch --parsable --dependency=afterok:${BUILD_JOB} "$SCRIPT_DIR/scalability.slurm")
echo "  Scalability job: $SCALE_JOB"

echo ""
echo "=== All jobs submitted ==="
echo "Build: $BUILD_JOB"
echo "Overhead: $OH_JOB (depends on build)"
echo "Compression: $COMP_JOB (depends on build)"
echo "Replay: $REP_JOB (depends on build)"
echo "Scalability: $SCALE_JOB (depends on build)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results will be in: $PROJECT_DIR/results/"
