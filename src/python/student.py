from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np


@dataclass
class OdeResult:
    t: np.ndarray
    y: np.ndarray
    status: int = 0
    success: bool = True
    message: str = "Integration completed successfully."


def solve_continuous_are(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Solve the continuous-time algebraic Riccati equation

        A^T P + P A - P B R^{-1} B^T P + Q = 0

    with a Hamiltonian stable-subspace construction.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]

    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must have the same number of rows as A.")
    if Q.ndim != 2 or Q.shape != (n, n):
        raise ValueError("Q must have the same shape as A.")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square.")

    Rinv = np.linalg.inv(R)

    h11 = A
    h12 = -(B @ Rinv @ B.T)
    h21 = -Q
    h22 = -A.T

    H = np.block([[h11, h12], [h21, h22]])

    eigvals, eigvecs = np.linalg.eig(H)

    stable_cols = [j for j, lam in enumerate(eigvals) if np.real(lam) < 0.0]
    if len(stable_cols) != n:
        raise np.linalg.LinAlgError(
            f"Hamiltonian matrix does not have exactly {n} stable eigenvalues."
        )

    W = eigvecs[:, stable_cols]
    X = W[:n, :]
    Y = W[n:, :]

    if np.linalg.matrix_rank(X) < n:
        raise np.linalg.LinAlgError("Stable subspace basis is not invertible in the upper block.")

    P = Y @ np.linalg.solve(X, np.eye(n))
    P = np.real_if_close(P, tol=1000)

    if np.iscomplexobj(P):
        max_imag = np.max(np.abs(P.imag))
        if max_imag > 1e-2:
            raise np.linalg.LinAlgError(
                f"Computed Riccati solution has significant imaginary part ({max_imag:.3e})."
            )
        P = P.real

    P = np.asarray(P, dtype=float)
    P = 0.5 * (P + P.T)
    return P


def _rk4_increment(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _advance_with_control(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_start: float,
    y_start: np.ndarray,
    t_stop: float,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """
    Adaptive RK4 on one interval [t_start, t_stop] using step-doubling
    for error estimation.
    """
    if t_stop == t_start:
        return y_start.copy()

    y = y_start.copy()
    t = float(t_start)

    sign = 1.0 if t_stop >= t_start else -1.0
    remaining_total = abs(t_stop - t_start)

    h = min(1e-3, remaining_total)

    while sign * (t_stop - t) > 0.0:
        h = min(h, abs(t_stop - t))
        hs = sign * h

        y_one = y + _rk4_increment(f, t, y, hs)

        y_half = y + _rk4_increment(f, t, y, 0.5 * hs)
        y_two = y_half + _rk4_increment(f, t + 0.5 * hs, y_half, 0.5 * hs)

        err_est = y_two - y_one
        denom = atol + rtol * np.maximum(np.abs(y), np.abs(y_two))
        ratio = np.max(np.abs(err_est) / denom)

        if ratio <= 1.0:
            t = t + hs
            y = y_two

            if ratio == 0.0:
                growth = 2.0
            else:
                growth = 0.9 * ratio ** (-0.2)
                growth = min(2.0, max(1.2, growth))
            h *= growth
        else:
            shrink = 0.9 * ratio ** (-0.2)
            shrink = max(0.1, shrink)
            h *= shrink

            if h < 1e-12:
                raise RuntimeError("Step size became too small during integration.")

    return y


def solve_ivp(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: Iterable[float] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    args: tuple = (),
    **kwargs,
) -> OdeResult:
    """
    Lightweight SciPy-like IVP solver.

    It integrates successively between output times using adaptive RK4.
    """
    _ = kwargs

    t0 = float(t_span[0])
    tf = float(t_span[1])

    y0 = np.asarray(y0, dtype=float)
    if y0.ndim != 1:
        raise ValueError("y0 must be a one-dimensional array.")

    if args:
        def wrapped_fun(t: float, y: np.ndarray) -> np.ndarray:
            return np.asarray(fun(t, y, *args), dtype=float)
    else:
        def wrapped_fun(t: float, y: np.ndarray) -> np.ndarray:
            return np.asarray(fun(t, y), dtype=float)

    if t_eval is None:
        t_grid = np.linspace(t0, tf, 1001)
    else:
        t_grid = np.asarray(list(t_eval), dtype=float)
        if t_grid.ndim != 1:
            raise ValueError("t_eval must be one-dimensional.")
        if t_grid.size == 0:
            raise ValueError("t_eval cannot be empty.")
        if not np.isclose(t_grid[0], t0) or not np.isclose(t_grid[-1], tf):
            raise ValueError("t_eval must begin at t_span[0] and end at t_span[1].")

    nstate = y0.size
    ntimes = t_grid.size

    Y = np.zeros((nstate, ntimes), dtype=float)
    Y[:, 0] = y0

    y_curr = y0.copy()
    t_curr = t_grid[0]

    for k in range(1, ntimes):
        t_next = t_grid[k]
        y_curr = _advance_with_control(
            wrapped_fun,
            t_curr,
            y_curr,
            t_next,
            rtol=rtol,
            atol=atol,
        )
        Y[:, k] = y_curr
        t_curr = t_next

    return OdeResult(t=t_grid, y=Y)
