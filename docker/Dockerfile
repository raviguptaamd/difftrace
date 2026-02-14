# DiffTrace: Provenance-Aware Reproducible Inference for Stochastic dLLMs
# Docker image for AMD MI300X with ROCm
#
# Build on compute node:
#   docker build -t difftrace:rocm6.2 -f docker/Dockerfile .
#
# Run:
#   docker run --rm --device=/dev/kfd --device=/dev/dri \
#     --group-add video --ipc=host --shm-size 64G \
#     -v /path/to/models:/models \
#     -v /path/to/output:/output \
#     difftrace:rocm6.2

FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Fix apt sources to use https
RUN sed -i 's/http/https/g' /etc/apt/sources.list

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    htop \
    numactl \
    && rm -rf /var/lib/apt/lists/*

# NOTE: For AMD internal cluster (useocpslog-002), use the cluster-specific
# Dockerfile.cluster which builds on top of pre-loaded images.

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install LLaDA model (clone and install)
RUN git clone https://github.com/ML-GSAI/LLaDA.git /workspace/LLaDA && \
    cd /workspace/LLaDA && \
    pip install -e . 2>/dev/null || pip install transformers accelerate

# Install DiffTrace
COPY . /workspace/difftrace
RUN cd /workspace/difftrace && pip install -e .

# Pre-download model weights (optional, can be mounted instead)
# RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('GSAI/LLaDA-8B-Instruct')"

# Default environment
ENV ROCM_PATH=/opt/rocm
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV PYTORCH_ROCM_ARCH="gfx942"
ENV OMP_NUM_THREADS=16

# Entry point
ENTRYPOINT ["/bin/bash"]
