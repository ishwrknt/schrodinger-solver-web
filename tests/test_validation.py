from schrodinger_solver_web.solver.eigensolver import solve_eigenstates
from schrodinger_solver_web.solver.grid import GridSpec, build_grid
from schrodinger_solver_web.solver.hamiltonian import assemble_hamiltonian
from schrodinger_solver_web.solver.potential_templates import InputMode, default_potential_definition, sample_potential
from schrodinger_solver_web.solver.validation import validate_postsolve, validate_presolve


def test_invalid_grid_configuration_is_rejected_before_solve():
    definition = default_potential_definition("harmonic_oscillator")
    errors = validate_presolve(GridSpec(x_min=1.0, x_max=0.0, num_points=10, n_modes=0), definition)
    assert errors


def test_invalid_analytic_expression_is_rejected_before_solve():
    definition = default_potential_definition("harmonic_oscillator")
    definition.mode = InputMode.ANALYTIC
    definition.label = "Analytic Potential"
    definition.template_id = None
    definition.expression = "__import__('os')"
    errors = validate_presolve(GridSpec(x_min=-2.0, x_max=2.0, num_points=201, n_modes=2), definition)
    assert any("Unsupported" in error for error in errors)


def test_postsolve_validation_accepts_well_resolved_harmonic_states():
    spec = GridSpec(x_min=-8.0, x_max=8.0, num_points=801, n_modes=2)
    definition = default_potential_definition("harmonic_oscillator")
    grid = build_grid(spec)
    states = solve_eigenstates(assemble_hamiltonian(grid, sample_potential(definition, grid.x)), 2)
    validations = validate_postsolve(spec, definition, states)
    assert validations
    assert all(validation.accepted for validation in validations)


def test_box_states_are_not_rejected_for_nonzero_interior_amplitude():
    import numpy as np

    spec = GridSpec(x_min=-0.5, x_max=0.5, num_points=401, n_modes=4)
    definition = default_potential_definition("infinite_square_well")
    grid = build_grid(spec)
    states = solve_eigenstates(assemble_hamiltonian(grid, sample_potential(definition, grid.x)), 4)
    assert abs(states[0].psi[1]) > 1e-4
    np.testing.assert_allclose([s.energy for s in states],
                               np.arange(1, 5) ** 2 * np.pi ** 2 / 2, rtol=1e-4)
    assert all(v.accepted for v in validate_postsolve(spec, definition, states))


def test_underresolved_box_states_still_fail_convergence():
    spec = GridSpec(x_min=-0.5, x_max=0.5, num_points=51, n_modes=12)
    definition = default_potential_definition("infinite_square_well")
    grid = build_grid(spec)
    states = solve_eigenstates(assemble_hamiltonian(grid, sample_potential(definition, grid.x)), 12)
    assert not validate_postsolve(spec, definition, states)[-1].accepted


def test_actual_boundary_violation_is_rejected():
    spec = GridSpec(x_min=-0.5, x_max=0.5, num_points=401, n_modes=1)
    definition = default_potential_definition("infinite_square_well")
    grid = build_grid(spec)
    states = solve_eigenstates(assemble_hamiltonian(grid, sample_potential(definition, grid.x)), 1)
    states[0].psi[0] = 0.1
    assert not validate_postsolve(spec, definition, states)[0].accepted
