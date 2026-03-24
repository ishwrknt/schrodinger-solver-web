from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    x_max: float
    num_points: int
    n_modes: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min


@dataclass(frozen=True)
class GridData:
    x: np.ndarray
    dx: float
    interior_x: np.ndarray
    spec: GridSpec


def validate_grid_spec(spec: GridSpec) -> list[str]:
    errors: list[str] = []
    if not np.isfinite([spec.x_min, spec.x_max]).all():
        errors.append("Domain bounds must be finite numbers.")
    if spec.x_max <= spec.x_min:
        errors.append("x_max must be greater than x_min.")
    if spec.width < 1e-3:
        errors.append("Domain width is too small for a stable finite-difference solve.")
    if spec.num_points < 51:
        errors.append("Grid size must be at least 51 points.")
    if spec.num_points > 10001:
        errors.append("Grid size must not exceed 10001 points in v1.")
    if spec.n_modes < 1:
        errors.append("At least one eigenmode must be requested.")
    max_modes = max(1, min(50, spec.num_points - 2))
    if spec.n_modes > max_modes:
        errors.append(f"Requested modes must be between 1 and {max_modes} for the selected grid.")
    return errors


def build_grid(spec: GridSpec) -> GridData:
    errors = validate_grid_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    x = np.linspace(spec.x_min, spec.x_max, spec.num_points, dtype=float)
    dx = float(x[1] - x[0])
    if dx <= 0.0:
        raise ValueError("Grid spacing must be positive.")
    return GridData(x=x, dx=dx, interior_x=x[1:-1], spec=spec)


def refined_grid_spec(spec: GridSpec) -> GridSpec:
    refined_points = min(10001, (spec.num_points * 2) - 1)
    return GridSpec(
        x_min=spec.x_min,
        x_max=spec.x_max,
        num_points=refined_points,
        n_modes=spec.n_modes,
    )

