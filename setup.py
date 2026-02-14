"""DiffTrace: Provenance-Aware Reproducible Inference for Stochastic dLLMs."""

from setuptools import setup, find_packages

setup(
    name="difftrace",
    version="0.1.0",
    description=(
        "Lightweight provenance framework for capturing and replaying "
        "stochastic denoising trajectories of diffusion language models"
    ),
    author="DiffTrace Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "zstandard>=0.21.0",
        "transformers>=4.40.0",
        "accelerate>=0.25.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-benchmark", "matplotlib", "pandas", "seaborn"],
        "experiment": [
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
            "seaborn>=0.12.0",
            "tabulate>=0.9.0",
        ],
    },
)
