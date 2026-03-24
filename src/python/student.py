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
    message: str = "The solver successfully reached the end of the integration interval."


def solve_continuous_are(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Solve the continuous-time algebraic Riccati equation

        A^T P + P A - P B R^{-1} B^T P + Q = 0

    by extracting the stable invariant subspace of the Hamiltonian matrix.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    n = A.shape[0]

    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must have the same number of rows as A.")
    if Q.ndim != 2 or Q.shape != (n, n):
        raise ValueError("Q must have the same shape as A.")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square.")

    R_inv = np.linalg.inv(R)

    H = np.block([
        [A, -B @ R_inv @ B.T],
        [-Q, -A.T],
    ])

    eigvals, eigvecs = np.linalg.eig(H)

    stable_cols = np.where(np.real(eigvals) < 0.0)[0]
    if stable_cols.size != n:
        raise np.linalg.LinAlgError(
            f"Expected {n} stable eigenvalues, found {stable_cols.size}."
        )

    Z = eigvecs[:, stable_cols]
    U = Z[:n, :]
    V = Z[n:, :]

    if np.linalg.matrix_rank(U) < n:
        raise np.linalg.LinAlgError("Stable invariant subspace is singular.")

    P = V @ np.linalg.solve(U, np.eye(n))
    P = np.real_if_close(P, tol=1000)

    if np.iscomplexobj(P):
        imag_size = np.max(np.abs(np.imag(P)))
        if imag_size > 1e-2:
            raise np.linalg.LinAlgError(
                f"Riccati solution has large imaginary part: {imag_size:.3e}"
            )
        P = np.real(P)

    P = np.asarray(P, dtype=float)
    P = 0.5 * (P + P.T)
    return P


def _rk4_step(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    k1 = np.asarray(fun(t, y), dtype=float)
    k2 = np.asarray(fun(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(fun(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(fun(t + h, y + h * k3), dtype=float)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_between_output_times(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    t1: float,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """
    Adaptive RK4 with step doubling on one segment [t0, t1].
    """
    if t1 == t0:
        return y0.copy()

    direction = 1.0 if t1 > t0 else -1.0
    total_length = abs(t1 - t0)

    t = float(t0)
    y = y0.copy()

    h = min(total_length, 1e-3)

    while direction * (t1 - t) > 0.0:
        h = min(h, abs(t1 - t))
        hs = direction * h

        y_full = _rk4_step(fun, t, y, hs)

        y_half = _rk4_step(fun, t, y, 0.5 * hs)
        y_two_half = _rk4_step(fun, t + 0.5 * hs, y_half, 0.5 * hs)

        err = y_two_half - y_full
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_two_half))
        err_norm = np.max(np.abs(err) / scale)

        if err_norm <= 1.0:
            t = t + hs
            y = y_two_half

            if err_norm == 0.0:
                growth = 2.0
            else:
                growth = 0.9 * err_norm ** (-0.2)
                growth = min(2.0, max(1.2, growth))
            h = min(total_length, h * growth)
        else:
            shrink = 0.9 * err_norm ** (-0.2)
            shrink = max(0.1, shrink)
            h = h * shrink
            if h < 1e-12:
                raise RuntimeError("Adaptive RK4 step size underflow.")

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
    Minimal SciPy-like IVP solver for this project.

    It advances adaptively between consecutive output times using RK4
    with step-doubling error control.
    """
    del kwargs

    t0 = float(t_span[0])
    tf = float(t_span[1])

    y0 = np.asarray(y0, dtype=float)
    if y0.ndim != 1:
        raise ValueError("y0 must be one-dimensional.")

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
            raise ValueError("t_eval must contain at least one time.")
        if abs(t_grid[0] - t0) > 1e-12 or abs(t_grid[-1] - tf) > 1e-12:
            raise ValueError("t_eval must start at t_span[0] and end at t_span[1].")

    n = y0.size
    m = t_grid.size
    y = np.zeros((n, m), dtype=float)
    y[:, 0] = y0

    current_t = t_grid[0]
    current_y = y0.copy()

    for j in range(m - 1):
        next_t = t_grid[j + 1]
        current_y = _integrate_between_output_times(
            wrapped_fun,
            current_t,
            current_y,
            next_t,
            rtol=rtol,
            atol=atol,
        )
        current_t = next_t
        y[:, j + 1] = current_y

    return OdeResult(t=t_grid, y=y)
