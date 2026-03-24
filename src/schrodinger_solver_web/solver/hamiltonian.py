from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schrodinger_solver_web.solver.grid import GridData


@dataclass(frozen=True)
class Hamiltonian1D:
    diagonal: np.ndarray
    off_diagonal: np.ndarray
    potential: np.ndarray
    grid: GridData

    def as_dense(self) -> np.ndarray:
        matrix = np.diag(self.diagonal)
        if self.off_diagonal.size:
            matrix += np.diag(self.off_diagonal, k=1)
            matrix += np.diag(self.off_diagonal, k=-1)
        return matrix


def assemble_hamiltonian(grid: GridData, potential: np.ndarray) -> Hamiltonian1D:
    if potential.shape != grid.x.shape:
        raise ValueError("Potential samples must match the full grid shape.")
    interior_potential = np.asarray(potential[1:-1], dtype=float)
    kinetic_diagonal = np.full(grid.interior_x.shape, 1.0 / (grid.dx**2), dtype=float)
    kinetic_off_diagonal = np.full(grid.interior_x.size - 1, -0.5 / (grid.dx**2), dtype=float)
    diagonal = kinetic_diagonal + interior_potential
    return Hamiltonian1D(
        diagonal=diagonal,
        off_diagonal=kinetic_off_diagonal,
        potential=np.asarray(potential, dtype=float),
        grid=grid,
    )

