"""
DiffTrace: Provenance-Aware Reproducible Inference for Stochastic dLLMs.

A lightweight provenance framework that captures the stochastic denoising
trajectory of diffusion language models during distributed inference.
"""

__version__ = "0.1.0"

from difftrace.core.trajectory import TrajectoryCapture, DenoiseStep
from difftrace.core.compressor import DifferentialCompressor
from difftrace.core.async_io import AsyncProvenanceWriter
from difftrace.core.store import ProvenanceStore, ProvenanceRecord
from difftrace.core.replay import ReplayEngine

__all__ = [
    "TrajectoryCapture",
    "DenoiseStep",
    "DifferentialCompressor",
    "AsyncProvenanceWriter",
    "ProvenanceStore",
    "ProvenanceRecord",
    "ReplayEngine",
]
