"""Inference layer — forward-backward algorithm and responsibilities."""

from hmm_core.inference.forward_backward import run_forward_backward, ForwardBackwardResult
from hmm_core.inference.responsibilities import compute_responsibilities

__all__ = ["run_forward_backward", "ForwardBackwardResult", "compute_responsibilities"]
