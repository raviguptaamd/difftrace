"""
LLaDA Integration Hook for DiffTrace.

Provides instrumentation for LLaDA (Large Language Diffusion Model) inference,
capturing the masked diffusion denoising trajectory at each timestep.

LLaDA uses a forward data masking process and reverse denoising process:
1. Start with prompt + fully masked generation region
2. At each denoising step t (from T down to 0):
   a. Model predicts token probabilities for all masked positions
   b. A schedule determines how many positions to unmask
   c. Top-confidence positions are unmasked with sampled tokens
3. Final output: fully unmasked sequence

DiffTrace hooks into step (2) to capture the trajectory.
"""

import time
import uuid
import logging
from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

from difftrace.core.trajectory import TrajectoryCapture, TrajectoryRecord
from difftrace.core.compressor import DifferentialCompressor
from difftrace.core.store import ProvenanceStore
from difftrace.utils.config import DiffTraceConfig, CaptureGranularity

logger = logging.getLogger(__name__)

# LLaDA mask token ID (typically the last token in vocabulary)
LLADA_MASK_TOKEN = 126336  # Default for LLaDA-8B


class LLaDADiffTraceHook:
    """
    DiffTrace hook for LLaDA inference.

    Wraps LLaDA's generate function to transparently capture
    the denoising trajectory with minimal overhead.
    """

    def __init__(
        self,
        config: DiffTraceConfig,
        mask_token_id: int = LLADA_MASK_TOKEN,
        store: Optional[ProvenanceStore] = None,
    ):
        self.config = config
        self.mask_token_id = mask_token_id
        self._capture = TrajectoryCapture(config.capture)
        self._compressor = DifferentialCompressor(config.compression)
        self._store = store

        # Performance tracking
        self._total_overhead_ns = 0
        self._num_requests = 0

    def generate(
        self,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        seq_length: int = 256,
        steps: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        cfg_scale: float = 0.0,
        tokenizer: Any = None,
        device: str = "cuda:0",
        request_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[TrajectoryRecord]]:
        """
        Run LLaDA inference with DiffTrace provenance capture.

        This implements the masked diffusion denoising loop with
        provenance hooks at each step.

        Args:
            model: LLaDA model
            prompt_ids: Tokenized prompt [batch_size, prompt_len]
            seq_length: Total sequence length (prompt + generation)
            steps: Number of denoising steps
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold
            cfg_scale: Classifier-free guidance scale (0 = disabled)
            tokenizer: Tokenizer for text decoding
            device: Target device
            request_id: Unique request identifier (auto-generated if None)

        Returns:
            (generated_ids, trajectory_record)
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        gen_len = seq_length - prompt_len

        # Build initial sequence: [prompt_tokens, MASK, MASK, ..., MASK]
        x = torch.full(
            (batch_size, seq_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        x[:, :prompt_len] = prompt_ids.to(device)

        # Prompt text for metadata
        prompt_text = ""
        if tokenizer is not None:
            try:
                prompt_text = tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=True)
            except Exception:
                pass

        # Begin trajectory capture
        trajectory = None
        if self.config.enabled:
            trajectory = self._capture.begin_trajectory(
                request_id=request_id,
                prompt_tokens=prompt_ids[0].cpu().numpy().astype(np.int32),
                prompt_text=prompt_text,
                model_name=getattr(model, "name_or_path", model.__class__.__name__),
                model_config={
                    "seq_length": seq_length,
                    "steps": steps,
                    "temperature": temperature,
                    "top_p": top_p,
                    "cfg_scale": cfg_scale,
                },
                num_diffusion_steps=steps,
                sequence_length=seq_length,
                temperature=temperature,
                top_p=top_p,
                device=device,
                world_size=self.config.distributed.world_size,
                rank=self.config.distributed.rank,
            )

        # === Masked Diffusion Denoising Loop ===
        model.eval()
        with torch.no_grad():
            for step_idx in range(steps):
                t = steps - step_idx  # Current timestep (T -> 1)

                # Get mask: True for positions that are still masked
                is_masked = (x == self.mask_token_id)

                # Count remaining masked positions
                num_masked = is_masked[:, prompt_len:].sum(dim=-1)

                # If nothing left to unmask, we're done
                if num_masked.max() == 0:
                    break

                # Forward pass: get logits for all positions
                logits = model(x).logits if hasattr(model(x), 'logits') else model(x)

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Classifier-free guidance (if applicable)
                if cfg_scale > 0:
                    # Create unconditional input (fully masked)
                    x_uncond = x.clone()
                    x_uncond[:, :prompt_len] = self.mask_token_id
                    logits_uncond = model(x_uncond).logits if hasattr(model(x_uncond), 'logits') else model(x_uncond)
                    logits = logits_uncond + cfg_scale * (logits - logits_uncond)

                # Sample tokens for masked positions
                probs = F.softmax(logits, dim=-1)

                # For each masked position, sample a token
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]),
                    num_samples=1,
                ).view(batch_size, seq_length)

                # Compute confidence: probability of sampled token
                confidence = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)

                # Determine which positions to unmask at this step
                # Schedule: linear unmasking (unmask proportional fraction)
                # At step t (from T to 1), unmask fraction = 1/t of remaining
                unmask_fraction = 1.0 / max(t, 1)
                num_to_unmask = (num_masked.float() * unmask_fraction).long().clamp(min=1)

                # Create per-sample unmask decisions based on confidence
                # For the generation region only
                gen_confidence = confidence[:, prompt_len:] * is_masked[:, prompt_len:].float()

                # Get top-k positions by confidence
                for b in range(batch_size):
                    k = min(int(num_to_unmask[b].item()), int(num_masked[b].item()))
                    if k == 0:
                        continue

                    # Top-k confident positions in generation region
                    _, top_indices = gen_confidence[b].topk(k)

                    # Map to full sequence positions
                    full_positions = top_indices + prompt_len
                    selected_tokens = sampled[b, full_positions]

                    # Unmask these positions
                    x[b, full_positions] = selected_tokens

                    # Capture this step in DiffTrace
                    if self.config.enabled and trajectory is not None:
                        step_logits = None
                        if self.config.capture.granularity == CaptureGranularity.FULL_LOGITS:
                            step_logits = logits[b]

                        self._capture.capture_step(
                            step_index=step_idx,
                            unmasked_positions=full_positions,
                            sampled_tokens=selected_tokens,
                            mask_state=(
                                (x[b] == self.mask_token_id)
                                if self.config.capture.granularity
                                in (CaptureGranularity.TOKENS_AND_MASKS,
                                    CaptureGranularity.FULL_LOGITS)
                                else None
                            ),
                            logits=step_logits,
                            confidence_scores=confidence[b, full_positions],
                            device=device,
                        )

        # Finalize trajectory
        if self.config.enabled and trajectory is not None:
            generated_text = ""
            if tokenizer is not None:
                try:
                    generated_text = tokenizer.decode(
                        x[0].tolist(), skip_special_tokens=True
                    )
                except Exception:
                    pass

            trajectory = self._capture.end_trajectory(
                generated_tokens=x[0],
                generated_text=generated_text,
            )

            # Store provenance
            if self._store is not None and trajectory is not None:
                self._store.store_trajectory(trajectory)

            self._num_requests += 1
            self._total_overhead_ns += int(self._capture.capture_overhead_ms * 1e6)

        return x, trajectory

    @property
    def avg_overhead_ms(self) -> float:
        """Average DiffTrace overhead per request in milliseconds."""
        if self._num_requests == 0:
            return 0.0
        return (self._total_overhead_ns / self._num_requests) / 1e6


def generate_with_difftrace(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    config: Optional[DiffTraceConfig] = None,
    seq_length: int = 256,
    steps: int = 64,
    temperature: float = 1.0,
    device: str = "cuda:0",
    store_path: Optional[str] = None,
) -> Tuple[str, Optional[TrajectoryRecord]]:
    """
    Convenience function: generate text with DiffTrace provenance.

    Args:
        model: LLaDA model
        tokenizer: Tokenizer
        prompt: Text prompt
        config: DiffTrace config (default: minimal)
        seq_length: Total sequence length
        steps: Denoising steps
        temperature: Sampling temperature
        device: Target device
        store_path: Path for provenance storage

    Returns:
        (generated_text, trajectory)
    """
    if config is None:
        config = DiffTraceConfig.minimal()

    store = None
    if store_path is not None:
        config.store.base_path = Path(store_path)
        store = ProvenanceStore(config)

    hook = LLaDADiffTraceHook(config=config, store=store)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"]

    # Generate
    output_ids, trajectory = hook.generate(
        model=model,
        prompt_ids=prompt_ids,
        seq_length=seq_length,
        steps=steps,
        temperature=temperature,
        tokenizer=tokenizer,
        device=device,
    )

    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    # Cleanup
    if store is not None:
        store.flush()

    return generated_text, trajectory
