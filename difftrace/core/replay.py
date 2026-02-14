"""
Replay Engine Module.

Enables deterministic replay and verification of dLLM inference
from stored provenance records. Supports two replay modes:

1. Token Replay: Reconstruct output from stored token selections
   (no model needed, instant replay)
2. Verified Replay: Re-run the model with stored RNG states and
   verify output matches (requires model, proves reproducibility)
"""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np

import torch

from difftrace.core.trajectory import TrajectoryCapture, TrajectoryRecord, DenoiseStep
from difftrace.core.compressor import (
    CompressedTrajectory,
    CompressedStep,
    DifferentialCompressor,
)
from difftrace.utils.config import CompressionConfig, CaptureGranularity

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Result of a replay operation."""
    request_id: str
    replay_mode: str  # "token" or "verified"
    success: bool
    generated_tokens: Optional[np.ndarray] = None
    generated_text: str = ""

    # Verification results
    token_match_rate: float = 0.0  # Fraction of tokens matching original
    exact_match: bool = False       # All tokens match
    mismatched_positions: List[int] = field(default_factory=list)

    # Timing
    replay_time_ms: float = 0.0
    original_time_ms: float = 0.0
    speedup: float = 0.0

    # Step-level verification
    step_match_rates: List[float] = field(default_factory=list)


class ReplayEngine:
    """
    Deterministic replay engine for dLLM inference.

    Supports reconstructing inference outputs from provenance records
    and verifying reproducibility by re-running inference with stored
    random states.
    """

    def __init__(self, compressor: Optional[DifferentialCompressor] = None):
        if compressor is None:
            compressor = DifferentialCompressor(CompressionConfig())
        self._compressor = compressor

    def token_replay(
        self,
        compressed: CompressedTrajectory,
        vocab_size: int = 0,
    ) -> ReplayResult:
        """
        Replay from stored token selections (no model needed).

        Reconstructs the full denoising trajectory from the differential
        provenance record by applying stored token selections step-by-step.
        """
        t_start = time.time_ns()
        request_id = compressed.request_id
        metadata = compressed.metadata

        # Reconstruct prompt tokens
        prompt_tokens = np.frombuffer(metadata["prompt_tokens"], dtype=np.int32)
        seq_len = metadata.get("sequence_length", 0)

        if seq_len == 0:
            logger.error(f"Cannot replay: sequence_length not stored for {request_id}")
            return ReplayResult(
                request_id=request_id,
                replay_mode="token",
                success=False,
            )

        # Initialize full sequence with mask tokens (use -1 as mask)
        MASK_TOKEN = -1
        sequence = np.full(seq_len, MASK_TOKEN, dtype=np.int32)
        prompt_len = len(prompt_tokens)
        sequence[:prompt_len] = prompt_tokens

        # Apply each step's token selections
        for cstep in compressed.steps:
            positions = self._compressor.decompress_array(
                cstep.positions_data, dtype=np.int32
            )
            tokens = self._compressor.decompress_array(
                cstep.tokens_data, dtype=np.int32
            )

            if len(positions) > 0 and len(tokens) > 0:
                for pos, tok in zip(positions, tokens):
                    if 0 <= pos < seq_len:
                        sequence[pos] = tok

        # Verify against stored final output
        generated_tokens = None
        exact_match = False
        token_match_rate = 0.0
        mismatched = []

        if "generated_tokens" in metadata:
            stored_output = np.frombuffer(
                metadata["generated_tokens"], dtype=np.int32
            )
            generated_tokens = stored_output

            # Compare replay result with stored output
            gen_region = sequence[prompt_len:]
            stored_gen = stored_output[prompt_len:] if len(stored_output) > prompt_len else stored_output

            min_len = min(len(gen_region), len(stored_gen))
            if min_len > 0:
                matches = (gen_region[:min_len] == stored_gen[:min_len])
                token_match_rate = float(matches.mean())
                exact_match = bool(matches.all())
                mismatched = list(np.where(~matches)[0])
        else:
            generated_tokens = sequence[prompt_len:]
            exact_match = True
            token_match_rate = 1.0

        t_end = time.time_ns()
        replay_time_ms = (t_end - t_start) / 1e6

        original_time_ms = 0.0
        if metadata.get("start_time_ns") and metadata.get("end_time_ns"):
            original_time_ms = (
                metadata["end_time_ns"] - metadata["start_time_ns"]
            ) / 1e6

        return ReplayResult(
            request_id=request_id,
            replay_mode="token",
            success=True,
            generated_tokens=generated_tokens,
            generated_text=metadata.get("generated_text", ""),
            token_match_rate=token_match_rate,
            exact_match=exact_match,
            mismatched_positions=mismatched,
            replay_time_ms=replay_time_ms,
            original_time_ms=original_time_ms,
            speedup=original_time_ms / replay_time_ms if replay_time_ms > 0 else 0,
        )

    def verified_replay(
        self,
        compressed: CompressedTrajectory,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = "cuda:0",
    ) -> ReplayResult:
        """
        Verified replay: re-run the model with stored RNG states.

        This proves that the model produces identical output given the
        same random state, establishing reproducibility.
        """
        t_start = time.time_ns()
        request_id = compressed.request_id
        metadata = compressed.metadata

        # Restore initial RNG state
        if metadata.get("initial_rng_state"):
            rng_state = DifferentialCompressor._deserialize_rng_state(
                metadata["initial_rng_state"]
            )
            TrajectoryCapture.restore_rng_state(rng_state, device)

        # Reconstruct prompt
        prompt_tokens = np.frombuffer(metadata["prompt_tokens"], dtype=np.int32)
        prompt_tensor = torch.from_numpy(prompt_tokens).unsqueeze(0).to(device)

        # Get generation config from metadata
        num_steps = metadata.get("num_diffusion_steps", 64)
        seq_len = metadata.get("sequence_length", 256)
        temperature = metadata.get("temperature", 1.0)

        # Run model inference with restored RNG state
        model.eval()
        with torch.no_grad():
            # This is model-specific; the hook layer handles the actual generation
            # For now, we provide the interface
            pass

        t_end = time.time_ns()

        # TODO: Complete verified replay with model-specific generation
        # This requires the model hook to implement generate_with_rng()
        return ReplayResult(
            request_id=request_id,
            replay_mode="verified",
            success=False,
            replay_time_ms=(t_end - t_start) / 1e6,
        )

    def compare_trajectories(
        self,
        traj_a: CompressedTrajectory,
        traj_b: CompressedTrajectory,
    ) -> Dict[str, Any]:
        """
        Compare two trajectories step-by-step.

        Useful for debugging non-determinism by identifying
        the first divergence point.
        """
        results = {
            "request_id_a": traj_a.request_id,
            "request_id_b": traj_b.request_id,
            "num_steps_a": len(traj_a.steps),
            "num_steps_b": len(traj_b.steps),
            "first_divergence_step": -1,
            "step_comparisons": [],
        }

        num_steps = min(len(traj_a.steps), len(traj_b.steps))
        for i in range(num_steps):
            step_a = traj_a.steps[i]
            step_b = traj_b.steps[i]

            pos_a = self._compressor.decompress_array(
                step_a.positions_data, dtype=np.int32
            )
            pos_b = self._compressor.decompress_array(
                step_b.positions_data, dtype=np.int32
            )
            tok_a = self._compressor.decompress_array(
                step_a.tokens_data, dtype=np.int32
            )
            tok_b = self._compressor.decompress_array(
                step_b.tokens_data, dtype=np.int32
            )

            positions_match = np.array_equal(pos_a, pos_b)
            tokens_match = np.array_equal(tok_a, tok_b)

            step_result = {
                "step": i,
                "positions_match": positions_match,
                "tokens_match": tokens_match,
                "num_unmasked_a": step_a.num_unmasked,
                "num_unmasked_b": step_b.num_unmasked,
            }

            if not (positions_match and tokens_match):
                if results["first_divergence_step"] == -1:
                    results["first_divergence_step"] = i

                # Detail the differences
                if not positions_match:
                    diff_pos = set(pos_a.tolist()) ^ set(pos_b.tolist())
                    step_result["position_diffs"] = list(diff_pos)[:20]

                if not tokens_match:
                    # Find token mismatches at shared positions
                    shared = set(pos_a.tolist()) & set(pos_b.tolist())
                    mismatches = []
                    pos_a_dict = dict(zip(pos_a.tolist(), tok_a.tolist()))
                    pos_b_dict = dict(zip(pos_b.tolist(), tok_b.tolist()))
                    for p in shared:
                        if pos_a_dict.get(p) != pos_b_dict.get(p):
                            mismatches.append({
                                "position": p,
                                "token_a": pos_a_dict[p],
                                "token_b": pos_b_dict[p],
                            })
                    step_result["token_mismatches"] = mismatches[:20]

            results["step_comparisons"].append(step_result)

        return results
