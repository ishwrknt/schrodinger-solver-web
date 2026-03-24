from __future__ import annotations

from dataclasses import dataclass

from schrodinger_solver_web.solver.eigensolver import Eigenstate, solve_eigenstates
from schrodinger_solver_web.solver.grid import GridSpec, build_grid, refined_grid_spec, validate_grid_spec
from schrodinger_solver_web.solver.hamiltonian import assemble_hamiltonian
from schrodinger_solver_web.solver.potential_templates import PotentialDefinition, sample_potential, validate_potential_definition


BOUNDARY_THRESHOLD = 1e-4
DRIFT_THRESHOLD = 1e-3


@dataclass(frozen=True)
class StateValidation:
    label: str
    accepted: bool
    boundary_left: float
    boundary_right: float
    eigenvalue_drift: float
    reasons: tuple[str, ...]


def validate_presolve(spec: GridSpec, definition: PotentialDefinition) -> list[str]:
    errors = validate_grid_spec(spec)
    errors.extend(validate_potential_definition(definition, spec.x_min, spec.x_max))
    return errors


def validate_postsolve(
    spec: GridSpec,
    definition: PotentialDefinition,
    states: list[Eigenstate],
) -> list[StateValidation]:
    refined_spec = refined_grid_spec(spec)
    refined_grid = build_grid(refined_spec)
    refined_potential = sample_potential(definition, refined_grid.x)
    refined_hamiltonian = assemble_hamiltonian(refined_grid, refined_potential)
    refined_states = solve_eigenstates(refined_hamiltonian, min(refined_spec.n_modes, len(states)))

    validations: list[StateValidation] = []
    for state, refined_state in zip(states, refined_states, strict=True):
        boundary_left = abs(float(state.psi[1]))
        boundary_right = abs(float(state.psi[-2]))
        drift = abs(float(state.energy - refined_state.energy))
        reasons: list[str] = []
        if boundary_left > BOUNDARY_THRESHOLD or boundary_right > BOUNDARY_THRESHOLD:
            reasons.append("Boundary amplitude threshold exceeded.")
        if drift > DRIFT_THRESHOLD:
            reasons.append("Refined-grid eigenvalue drift threshold exceeded.")
        validations.append(
            StateValidation(
                label=state.label,
                accepted=(len(reasons) == 0),
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                eigenvalue_drift=drift,
                reasons=tuple(reasons),
            )
        )
    return validations
