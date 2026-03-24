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
