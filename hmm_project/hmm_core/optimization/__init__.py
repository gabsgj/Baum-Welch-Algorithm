"""Optimization layer — Baum-Welch EM update and convergence."""

from hmm_core.optimization.baum_welch_step import baum_welch_update
from hmm_core.optimization.convergence import check_convergence

__all__ = ["baum_welch_update", "check_convergence"]
