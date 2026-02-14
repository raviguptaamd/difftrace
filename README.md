# DiffTrace: Provenance-Aware Reproducible Inference for Stochastic Diffusion Language Models

**DiffTrace** is a lightweight provenance framework that captures the stochastic denoising trajectory of diffusion language models (dLLMs) during inference, enabling deterministic replay and reproducibility.

## Key Results (LLaDA-8B-Instruct on AMD MI300X)

| Metric | Value |
|---|---|
| Overhead (tokens+masks) | **< 1.1%** |
| Compression (diff + zstd) | **2.4x** |
| Replay accuracy | **100% exact match** |
| Replay time (L=128, T=64) | **0.36 ms** (3,700x faster than inference) |

## Architecture

DiffTrace hooks into the dLLM denoising loop to capture:
- **Token selections**: which positions are unmasked and what tokens are chosen at each step
- **Mask states**: the full binary mask after each denoising step
- **Logits** (optional): complete model output for debugging

Captured data is differentially encoded (only deltas between steps), compressed with zstd, and written asynchronously to avoid blocking inference.

## Installation

```bash
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 1.13+
- zstandard
- numpy

For LLaDA model experiments:
- transformers
- accelerate
- AMD MI300X GPU with ROCm 7.2+

## Quick Start

```python
import numpy as np
from difftrace.engine import DiffTraceEngine
from difftrace.core.trajectory import TrajectoryCapture
from difftrace.utils.config import DiffTraceConfig, CaptureGranularity

# Initialize engine
config = DiffTraceConfig()
config.capture.granularity = CaptureGranularity.TOKENS_AND_MASKS
engine = DiffTraceEngine(config)

# Create a capture session
capture = TrajectoryCapture(config.capture)
trajectory = capture.begin_trajectory(
    request_id="my_request_id",
    prompt_tokens=np.array(prompt_token_ids, dtype=np.int32),
    model_name="LLaDA-8B",
    num_diffusion_steps=128,
    sequence_length=256,
    device="cuda:0",
)

# In your denoising loop, after each step:
capture.capture_step(
    step_index=step_idx,
    unmasked_positions=positions_tensor,  # torch.Tensor
    sampled_tokens=tokens_tensor,         # torch.Tensor
    mask_state=mask_tensor,               # torch.Tensor (optional)
)

# Finalize and store
trajectory = capture.end_trajectory(generated_tokens=output_tensor)
engine.store_trajectory(trajectory)

# Replay (no model needed -- instant token reconstruction)
result = engine.replay("my_request_id")
assert result.exact_match  # 100% match guaranteed
```

## Experiments

### Real Model Benchmark (LLaDA-8B-Instruct)
```bash
python experiments/bench_real_llada.py \
    --model-path /path/to/LLaDA-8B-Instruct \
    --device cuda:0 \
    --gen-lengths 64 128 256 \
    --steps 32 64 128
```

### Component Benchmarks
```bash
python experiments/bench_overhead.py      # Latency overhead
python experiments/bench_compression.py   # Compression ratios
python experiments/bench_replay.py        # Replay accuracy
python experiments/bench_scalability.py   # Multi-GPU scaling
```

### End-to-End Tests
```bash
python experiments/test_e2e.py
```

## Docker (AMD MI300X)

```bash
docker build -f docker/Dockerfile.cluster -t difftrace:latest .
docker run --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size 64G \
    difftrace:latest python experiments/bench_real_llada.py
```

## Paper

See `paper/` for the HPAI4S'26 (IPDPS 2026) workshop paper:

> **DiffTrace: Provenance-Aware Reproducible Inference for Stochastic Diffusion Language Models**
> Ravi Gupta (AMD)

## Project Structure

```
difftrace/
├── difftrace/           # Core library
│   ├── core/            # Trajectory, compressor, async I/O, store, replay
│   ├── hooks/           # Model-specific hooks (LLaDA)
│   ├── utils/           # Configuration
│   └── engine.py        # Main API
├── experiments/         # Benchmarks and tests
│   ├── bench_real_llada.py    # Real model benchmark
│   ├── bench_overhead.py      # Overhead microbenchmark
│   ├── bench_compression.py   # Compression benchmark
│   ├── bench_replay.py        # Replay benchmark
│   ├── bench_scalability.py   # Multi-GPU scalability
│   └── test_e2e.py            # End-to-end tests
├── paper/               # LaTeX paper source
├── docker/              # Dockerfiles
├── results/             # Benchmark results (JSON)
└── analysis/            # Plotting scripts
```

## License

Apache 2.0

## Citation

```bibtex
@inproceedings{gupta2026difftrace,
  title={DiffTrace: Provenance-Aware Reproducible Inference for Stochastic Diffusion Language Models},
  author={Gupta, Ravi},
  booktitle={HPAI4S Workshop at IEEE IPDPS},
  year={2026}
}
```
