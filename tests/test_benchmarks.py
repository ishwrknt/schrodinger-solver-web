import numpy as np

from schrodinger_solver_web.solver.benchmarks import harmonic_oscillator_energies, infinite_square_well_energies
from schrodinger_solver_web.solver.eigensolver import solve_eigenstates
from schrodinger_solver_web.solver.grid import GridSpec, build_grid
from schrodinger_solver_web.solver.hamiltonian import assemble_hamiltonian
from schrodinger_solver_web.solver.potential_templates import default_potential_definition, sample_potential


def test_infinite_square_well_numerics_track_analytic_energies():
    grid = build_grid(GridSpec(x_min=0.0, x_max=1.0, num_points=801, n_modes=3))
    definition = default_potential_definition("infinite_square_well")
    definition.parameters.update({"center": 0.5, "width": 1.0, "wall_height": 1e8})
    potential = sample_potential(definition, grid.x)
    states = solve_eigenstates(assemble_hamiltonian(grid, potential), 3)
    analytic = infinite_square_well_energies(1.0, 3)
    assert np.allclose([state.energy for state in states], analytic, rtol=0.01)


def test_harmonic_oscillator_analytic_formula_is_consistent():
    energies = harmonic_oscillator_energies(k=4.0, offset=1.5, n_modes=3)
    assert np.allclose(energies, np.array([2.5, 4.5, 6.5]))

