import numpy as np

from schrodinger_solver_web.solver.grid import GridSpec, build_grid
from schrodinger_solver_web.solver.hamiltonian import assemble_hamiltonian


def test_hamiltonian_dimensions_and_structure():
    grid = build_grid(GridSpec(x_min=-1.0, x_max=1.0, num_points=101, n_modes=3))
    hamiltonian = assemble_hamiltonian(grid, np.zeros_like(grid.x))
    dense = hamiltonian.as_dense()
    assert dense.shape == (99, 99)
    assert np.allclose(np.diag(dense, 1), np.full(98, -0.5 / (grid.dx**2)))
    assert np.allclose(np.diag(dense), np.full(99, 1.0 / (grid.dx**2)))

