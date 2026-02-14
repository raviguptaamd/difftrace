"""
Trajectory Capture Module.

Captures the stochastic denoising trajectory of diffusion language models
at each timestep during inference. This is the core data structure and
capture logic for DiffTrace.
"""

import time
import struct
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

import torch

from difftrace.utils.config import CaptureConfig, CaptureGranularity


@dataclass
class DenoiseStep:
    """
    A single denoising step in the diffusion trajectory.

    Captures the state transition from step t to step t-1.
    In masked diffusion models (e.g., LLaDA), this records which positions
    were unmasked and what tokens were sampled.
    """
    step_index: int               # Denoising step index (T -> 0)
    timestamp_ns: int = 0         # Wall-clock timestamp (nanoseconds)

    # Token-level state (always captured)
    unmasked_positions: Optional[np.ndarray] = None   # int32: positions unmasked at this step
    sampled_tokens: Optional[np.ndarray] = None       # int32: token IDs sampled at unmasked positions

    # Mask state (captured at TOKENS_AND_MASKS granularity and above)
    mask_state: Optional[np.ndarray] = None           # bool: full mask (True=masked) after this step

    # Full logits (captured at FULL_LOGITS granularity)
    logits: Optional[np.ndarray] = None               # float16/float32: model output logits

    # Random state for deterministic replay
    rng_state: Optional[Dict[str, Any]] = None        # PyTorch RNG state

    # Per-step metadata
    num_unmasked: int = 0         # Number of positions unmasked at this step
    confidence_scores: Optional[np.ndarray] = None  # float32: confidence of sampled tokens

    def size_bytes(self) -> int:
        """Compute approximate memory footprint of this step."""
        total = struct.calcsize("i") * 2 + struct.calcsize("q")  # step_index, num_unmasked, timestamp
        if self.unmasked_positions is not None:
            total += self.unmasked_positions.nbytes
        if self.sampled_tokens is not None:
            total += self.sampled_tokens.nbytes
        if self.mask_state is not None:
            total += self.mask_state.nbytes
        if self.logits is not None:
            total += self.logits.nbytes
        if self.confidence_scores is not None:
            total += self.confidence_scores.nbytes
        if self.rng_state is not None:
            # Approximate: RNG state is typically ~5KB
            total += 5120
        return total


@dataclass
class TrajectoryRecord:
    """
    Complete denoising trajectory for a single inference request.
    """
    request_id: str
    prompt_tokens: np.ndarray         # int32: tokenized prompt
    prompt_text: str = ""
    generated_tokens: Optional[np.ndarray] = None  # int32: final generated sequence
    generated_text: str = ""

    # Model metadata
    model_name: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Inference parameters
    num_diffusion_steps: int = 0
    sequence_length: int = 0
    temperature: float = 1.0
    top_p: float = 1.0

    # Hardware/environment metadata
    device: str = ""
    gpu_name: str = ""
    world_size: int = 1
    rank: int = 0

    # Initial random state (for full replay)
    initial_rng_state: Optional[Dict[str, Any]] = None

    # The trajectory itself
    steps: List[DenoiseStep] = field(default_factory=list)

    # Timing
    start_time_ns: int = 0
    end_time_ns: int = 0

    @property
    def total_time_ms(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1e6

    @property
    def num_steps_captured(self) -> int:
        return len(self.steps)

    def total_size_bytes(self) -> int:
        """Total memory footprint of this trajectory."""
        base = self.prompt_tokens.nbytes
        if self.generated_tokens is not None:
            base += self.generated_tokens.nbytes
        base += sum(s.size_bytes() for s in self.steps)
        return base


class TrajectoryCapture:
    """
    Main trajectory capture engine.

    Hooks into the dLLM denoising loop to capture state at each timestep.
    Designed to be model-agnostic with specific hooks for different backends.
    """

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._active_trajectory: Optional[TrajectoryRecord] = None
        self._step_counter: int = 0
        self._capture_overhead_ns: int = 0

    def begin_trajectory(
        self,
        request_id: str,
        prompt_tokens: np.ndarray,
        prompt_text: str = "",
        model_name: str = "",
        model_config: Optional[Dict[str, Any]] = None,
        num_diffusion_steps: int = 0,
        sequence_length: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        device: str = "",
        world_size: int = 1,
        rank: int = 0,
    ) -> TrajectoryRecord:
        """Begin capturing a new inference trajectory."""
        self._step_counter = 0
        self._capture_overhead_ns = 0

        # Capture initial RNG state
        initial_rng = None
        if self.config.capture_random_state:
            initial_rng = self._capture_rng_state(device)

        gpu_name = ""
        if device.startswith("cuda") or device.startswith("hip"):
            try:
                idx = int(device.split(":")[-1]) if ":" in device else 0
                gpu_name = torch.cuda.get_device_name(idx)
            except Exception:
                gpu_name = "unknown"

        self._active_trajectory = TrajectoryRecord(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_text,
            model_name=model_name,
            model_config=model_config or {},
            num_diffusion_steps=num_diffusion_steps,
            sequence_length=sequence_length,
            temperature=temperature,
            top_p=top_p,
            device=device,
            gpu_name=gpu_name,
            world_size=world_size,
            rank=rank,
            initial_rng_state=initial_rng,
            start_time_ns=time.time_ns(),
        )
        return self._active_trajectory

    def capture_step(
        self,
        step_index: int,
        unmasked_positions: torch.Tensor,
        sampled_tokens: torch.Tensor,
        mask_state: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        device: str = "",
    ) -> Optional[DenoiseStep]:
        """
        Capture a single denoising step.

        This is called at each step of the diffusion denoising loop.
        Designed to be as lightweight as possible on the critical path.
        """
        if self._active_trajectory is None:
            return None

        # Skip steps based on capture frequency
        self._step_counter += 1
        if self._step_counter % self.config.capture_every_n_steps != 0:
            return None

        t_start = time.time_ns()

        # Build the step record - move tensors to CPU + numpy asynchronously
        step = DenoiseStep(
            step_index=step_index,
            timestamp_ns=time.time_ns() if self.config.capture_timestamps else 0,
            num_unmasked=unmasked_positions.numel(),
        )

        # Always capture token-level changes (minimal overhead)
        step.unmasked_positions = unmasked_positions.detach().cpu().numpy().astype(np.int32)
        step.sampled_tokens = sampled_tokens.detach().cpu().numpy().astype(np.int32)

        # Capture mask state if configured
        if (self.config.granularity in (
                CaptureGranularity.TOKENS_AND_MASKS,
                CaptureGranularity.FULL_LOGITS)
                and mask_state is not None):
            step.mask_state = mask_state.detach().cpu().numpy()

        # Capture full logits if configured
        if (self.config.granularity == CaptureGranularity.FULL_LOGITS
                and logits is not None):
            # Use float16 to halve memory
            step.logits = logits.detach().cpu().half().numpy()

        # Capture confidence scores if available
        if confidence_scores is not None:
            step.confidence_scores = confidence_scores.detach().cpu().numpy().astype(np.float32)

        # Capture RNG state
        if self.config.capture_random_state:
            step.rng_state = self._capture_rng_state(device)

        self._active_trajectory.steps.append(step)

        t_end = time.time_ns()
        self._capture_overhead_ns += (t_end - t_start)

        return step

    def end_trajectory(
        self,
        generated_tokens: Optional[torch.Tensor] = None,
        generated_text: str = "",
    ) -> Optional[TrajectoryRecord]:
        """Finalize the current trajectory capture."""
        if self._active_trajectory is None:
            return None

        self._active_trajectory.end_time_ns = time.time_ns()
        if generated_tokens is not None:
            self._active_trajectory.generated_tokens = (
                generated_tokens.detach().cpu().numpy().astype(np.int32)
            )
        self._active_trajectory.generated_text = generated_text

        trajectory = self._active_trajectory
        self._active_trajectory = None
        return trajectory

    @property
    def capture_overhead_ms(self) -> float:
        """Total capture overhead in milliseconds."""
        return self._capture_overhead_ns / 1e6

    @staticmethod
    def _capture_rng_state(device: str = "") -> Dict[str, Any]:
        """Capture current RNG state for replay."""
        state = {
            "python_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state().numpy().tobytes(),
        }
        if torch.cuda.is_available() and (
            device.startswith("cuda") or device.startswith("hip")
        ):
            try:
                idx = int(device.split(":")[-1]) if ":" in device else 0
                state["cuda_rng"] = torch.cuda.get_rng_state(idx).numpy().tobytes()
            except Exception:
                pass
        return state

    @staticmethod
    def restore_rng_state(state: Dict[str, Any], device: str = "") -> None:
        """Restore RNG state from a captured state dict."""
        if "python_rng" in state:
            np.random.set_state(state["python_rng"])
        if "torch_rng" in state:
            torch.random.set_rng_state(
                torch.from_numpy(np.frombuffer(state["torch_rng"], dtype=np.uint8))
            )
        if "cuda_rng" in state and torch.cuda.is_available():
            try:
                idx = int(device.split(":")[-1]) if ":" in device else 0
                torch.cuda.set_rng_state(
                    torch.from_numpy(np.frombuffer(state["cuda_rng"], dtype=np.uint8)),
                    device=idx,
                )
            except Exception:
                pass
