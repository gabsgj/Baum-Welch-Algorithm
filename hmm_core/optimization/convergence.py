"""
convergence.py
==============
EM convergence criterion.

Checks whether the absolute change in log-likelihood between
consecutive iterations is below a user-specified tolerance,
or whether the maximum iteration count has been reached.
"""

from __future__ import annotations


def check_convergence(
    ll_new: float,
    ll_old: float,
    tolerance: float,
    iteration: int,
    max_iterations: int,
) -> bool:
    """Return True if the EM loop should stop.

    Parameters
    ----------
    ll_new : float
        Log-likelihood after the current M-step.
    ll_old : float
        Log-likelihood from the previous iteration.
    tolerance : float
        Absolute convergence threshold.
    iteration : int
        Current (1-based) iteration number.
    max_iterations : int
        Hard upper bound on iterations.

    Returns
    -------
    bool
        ``True`` → stop;  ``False`` → continue.
    """
    if iteration >= max_iterations:
        return True
    if abs(ll_new - ll_old) < tolerance:
        return True
    return False
