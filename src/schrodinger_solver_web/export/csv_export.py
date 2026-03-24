from __future__ import annotations

import csv
import io

import numpy as np

from schrodinger_solver_web.solver.eigensolver import Eigenstate
from schrodinger_solver_web.solver.grid import GridSpec
from schrodinger_solver_web.solver.potential_templates import PotentialDefinition
from schrodinger_solver_web.solver.validation import StateValidation


def export_csv(
    grid_spec: GridSpec,
    definition: PotentialDefinition,
    x: np.ndarray,
    potential: np.ndarray,
    states: list[Eigenstate],
    validations: list[StateValidation],
) -> str:
    buffer = io.StringIO()
    metadata_writer = csv.writer(buffer)
    metadata = _metadata_rows(grid_spec, definition)
    for row in metadata:
        metadata_writer.writerow([f"# {row[0]}", row[1]])
    metadata_writer.writerow([])

    headers = ["x", "V"]
    for state in states:
        headers.extend([f"{state.label}_psi", f"{state.label}_probability"])
    headers.extend(["accepted_states", "rejected_states"])
    table_writer = csv.DictWriter(buffer, fieldnames=headers)
    table_writer.writeheader()

    accepted_labels = "|".join(validation.label for validation in validations if validation.accepted)
    rejected_labels = "|".join(validation.label for validation in validations if not validation.accepted)
    for index, position in enumerate(x):
        row = {
            "x": float(position),
            "V": float(potential[index]),
            "accepted_states": accepted_labels,
            "rejected_states": rejected_labels,
        }
        for state in states:
            row[f"{state.label}_psi"] = float(state.psi[index])
            row[f"{state.label}_probability"] = float(state.probability_density[index])
        table_writer.writerow(row)
    return buffer.getvalue()


def _metadata_rows(grid_spec: GridSpec, definition: PotentialDefinition) -> list[tuple[str, str]]:
    rows = [
        ("input_mode", definition.mode.value),
        ("potential_label", definition.label),
        ("template_id", definition.template_id or ""),
        ("expression", definition.expression or ""),
        ("domain", f"[{grid_spec.x_min}, {grid_spec.x_max}]"),
        ("grid_points", str(grid_spec.num_points)),
        ("requested_modes", str(grid_spec.n_modes)),
        ("unit_convention", "dimensionless (hbar=1, m=1)"),
    ]
    if definition.parameters:
        rows.append(("parameters", repr(definition.parameters)))
    if definition.segments:
        rows.append(("segments", repr([(s.start, s.end, s.value) for s in definition.segments])))
    return rows

