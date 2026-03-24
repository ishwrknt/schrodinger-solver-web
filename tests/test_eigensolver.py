import numpy as np

from schrodinger_solver_web.solver.eigensolver import solve_eigenstates
from schrodinger_solver_web.solver.grid import GridSpec, build_grid
from schrodinger_solver_web.solver.hamiltonian import assemble_hamiltonian


def test_eigensolver_returns_sorted_normalized_states():
    grid = build_grid(GridSpec(x_min=0.0, x_max=1.0, num_points=401, n_modes=3))
    potential = np.zeros_like(grid.x)
    states = solve_eigenstates(assemble_hamiltonian(grid, potential), 3)
    energies = [state.energy for state in states]
    assert energies == sorted(energies)
    for state in states:
        integral = np.trapezoid(state.probability_density, grid.x) if hasattr(np, "trapezoid") else np.trapz(state.probability_density, grid.x)
        assert np.isclose(integral, 1.0, atol=1e-6)
        assert state.label.startswith("n=")
