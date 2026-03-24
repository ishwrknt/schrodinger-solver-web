from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from schrodinger_solver_web.solver.hamiltonian import Hamiltonian1D


@dataclass(frozen=True)
class Eigenstate:
    index: int
    label: str
    energy: float
    psi: np.ndarray
    probability_density: np.ndarray


def solve_eigenstates(hamiltonian: Hamiltonian1D, n_modes: int) -> list[Eigenstate]:
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1.")
    if n_modes > hamiltonian.diagonal.size:
        raise ValueError("n_modes exceeds the number of solvable interior states.")
    eigenvalues, eigenvectors = eigh_tridiagonal(
        hamiltonian.diagonal,
        hamiltonian.off_diagonal,
        select="i",
        select_range=(0, n_modes - 1),
    )
    states: list[Eigenstate] = []
    for state_index in range(n_modes):
        full_wavefunction = np.zeros_like(hamiltonian.grid.x, dtype=float)
        full_wavefunction[1:-1] = eigenvectors[:, state_index]
        normalization = np.sqrt(_integrate(np.abs(full_wavefunction) ** 2, hamiltonian.grid.x))
        if normalization <= 0.0:
            raise ValueError("Encountered a non-normalizable eigenstate.")
        full_wavefunction = full_wavefunction / normalization
        states.append(
            Eigenstate(
                index=state_index + 1,
                label=f"n={state_index + 1}",
                energy=float(eigenvalues[state_index]),
                psi=full_wavefunction,
                probability_density=np.abs(full_wavefunction) ** 2,
            )
        )
    return states


def _integrate(values: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x))
    return float(np.trapz(values, x))
