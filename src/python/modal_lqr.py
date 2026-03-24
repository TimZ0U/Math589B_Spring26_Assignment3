"""Modal LQR control for a vibrating square membrane.

This module builds a truncated modal model for the unit square wave
equation with one localized actuator, designs an LQR controller,
simulates the closed-loop dynamics, and reconstructs membrane shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from .student import solve_ivp, solve_continuous_are
except ImportError:
    from student import solve_ivp, solve_continuous_are

Mode = Tuple[int, int]


@dataclass
class MembraneModel:
    c: float
    M: int
    x0: float
    y0: float
    gamma: float
    modes: List[Mode]
    omegas_sq: np.ndarray
    beta: np.ndarray
    A: np.ndarray
    B: np.ndarray


def build_modes(M: int) -> List[Mode]:
    return [(m, n) for m in range(1, M + 1) for n in range(1, M + 1)]


def square_eigenfunction(m: int, n: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)


def square_eigenvalue(m: int, n: int) -> float:
    return np.pi**2 * (m * m + n * n)


def point_coupling(m: int, n: int, x0: float, y0: float) -> float:
    return float(square_eigenfunction(m, n, np.asarray(x0), np.asarray(y0)))


def gaussian_patch(x: np.ndarray, y: np.ndarray, x0: float, y0: float, sigma: float) -> np.ndarray:
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))
    area = np.trapz(np.trapz(g, x=y[0, :], axis=1), x=x[:, 0], axis=0)
    return g / area


def patch_coupling(m: int, n: int, x0: float, y0: float, sigma: float = 0.06, ng: int = 121) -> float:
    grid = np.linspace(0.0, 1.0, ng)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    psi = gaussian_patch(X, Y, x0, y0, sigma)
    phi = square_eigenfunction(m, n, X, Y)
    inner = np.trapz(psi * phi, x=grid, axis=1)
    return float(np.trapz(inner, x=grid))


def build_model(
    M: int = 6,
    c: float = 1.0,
    x0: float = 0.37,
    y0: float = 0.61,
    actuator: str = "point",
    sigma: float = 0.06,
    gamma: float = 0.0,
) -> MembraneModel:
    modes = build_modes(M)
    N = len(modes)

    lambdas = np.array([square_eigenvalue(m, n) for m, n in modes], dtype=float)
    omegas_sq = (c * c) * lambdas

    if actuator == "point":
        beta = np.array([point_coupling(m, n, x0, y0) for m, n in modes], dtype=float)
    elif actuator == "patch":
        beta = np.array([patch_coupling(m, n, x0, y0, sigma=sigma) for m, n in modes], dtype=float)
    else:
        raise ValueError("actuator must be either 'point' or 'patch'.")

    stiffness = np.diag(omegas_sq)
    damping = gamma * np.eye(N)

    A = np.block([
        [np.zeros((N, N)), np.eye(N)],
        [-stiffness, -damping],
    ])

    B = np.vstack([
        np.zeros((N, 1)),
        beta.reshape(N, 1),
    ])

    return MembraneModel(
        c=c,
        M=M,
        x0=x0,
        y0=y0,
        gamma=gamma,
        modes=modes,
        omegas_sq=omegas_sq,
        beta=beta,
        A=A,
        B=B,
    )


def build_lqr(
    model: MembraneModel,
    alpha: float = 1.0,
    beta_v: float = 1.0,
    R: float = 5e-2,
):
    N = len(model.modes)

    Q = np.block([
        [alpha * np.diag(model.omegas_sq), np.zeros((N, N))],
        [np.zeros((N, N)), beta_v * np.eye(N)],
    ])
    Rmat = np.array([[R]], dtype=float)

    P = solve_continuous_are(model.A, model.B, Q, Rmat)
    K = np.linalg.solve(Rmat, model.B.T @ P)
    return Q, Rmat, P, K


def initial_state(
    model: MembraneModel,
    excited_modes: Sequence[Tuple[Mode, float]],
    excited_velocities: Sequence[Tuple[Mode, float]] | None = None,
) -> np.ndarray:
    N = len(model.modes)
    q0 = np.zeros(N, dtype=float)
    p0 = np.zeros(N, dtype=float)

    mode_to_index = {mode: i for i, mode in enumerate(model.modes)}

    for mode, amp in excited_modes:
        q0[mode_to_index[mode]] = amp

    if excited_velocities is not None:
        for mode, amp in excited_velocities:
            p0[mode_to_index[mode]] = amp

    return np.concatenate([q0, p0])


def simulate_closed_loop(
    model: MembraneModel,
    K: np.ndarray,
    x_init: np.ndarray,
    T: float = 6.0,
    nt: int = 800,
):
    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        u = float(-(K @ x).item())
        return model.A @ x + model.B[:, 0] * u

    t_eval = np.linspace(0.0, T, nt)
    sol = solve_ivp(rhs, (0.0, T), x_init, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    controls = np.array([float(-(K @ sol.y[:, j]).item()) for j in range(sol.y.shape[1])], dtype=float)
    return sol.t, sol.y, controls


def simulate_open_loop(
    model: MembraneModel,
    x_init: np.ndarray,
    T: float = 6.0,
    nt: int = 800,
):
    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        return model.A @ x

    t_eval = np.linspace(0.0, T, nt)
    sol = solve_ivp(rhs, (0.0, T), x_init, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def compute_energy(model: MembraneModel, y: np.ndarray) -> np.ndarray:
    N = len(model.modes)
    q = y[:N, :]
    p = y[N:, :]
    return 0.5 * np.sum(p * p + model.omegas_sq[:, None] * q * q, axis=0)


def reconstruct_field(model: MembraneModel, q: np.ndarray, grid_size: int = 81):
    grid = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    U = np.zeros_like(X)

    for coeff, (m, n) in zip(q, model.modes):
        U += coeff * square_eigenfunction(m, n, X, Y)

    return X, Y, U


def reconstruct_time_series(
    model: MembraneModel,
    y: np.ndarray,
    time_indices: Iterable[int],
    grid_size: int = 81,
):
    N = len(model.modes)
    frames = []

    for j in time_indices:
        _, _, U = reconstruct_field(model, y[:N, j], grid_size=grid_size)
        frames.append(U)

    return frames


def summarize_couplings(model: MembraneModel, count: int = 12) -> str:
    lines = [
        f"Actuator location: ({model.x0:.3f}, {model.y0:.3f})",
        f"Viscosity gamma: {model.gamma:.6f}",
        "First modal couplings beta_(m,n):",
    ]
    for (m, n), b in list(zip(model.modes, model.beta))[:count]:
        lines.append(f"  mode ({m},{n}): {b:+.6f}")
    return "\n".join(lines)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def demo_configuration() -> Tuple[MembraneModel, np.ndarray]:
    model = build_model(M=6, x0=0.37, y0=0.61, actuator="point", gamma=0.0)
    x0 = initial_state(
        model,
        excited_modes=[((1, 1), 0.8), ((2, 1), 0.3), ((1, 3), -0.25)],
        excited_velocities=[((1, 2), 0.15)],
    )
    return model, x0
