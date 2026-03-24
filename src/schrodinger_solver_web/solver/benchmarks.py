from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_hermite, factorial

from schrodinger_solver_web.solver.eigensolver import Eigenstate
from schrodinger_solver_web.solver.potential_templates import PotentialDefinition


@dataclass(frozen=True)
class BenchmarkRow:
    label: str
    numerical_energy: float
    analytic_energy: float
    energy_delta: float


@dataclass(frozen=True)
class BenchmarkComparison:
    benchmark_name: str
    rows: list[BenchmarkRow]
    analytic_wavefunctions: dict[str, np.ndarray]


def infinite_square_well_energies(width: float, n_modes: int) -> np.ndarray:
    n = np.arange(1, n_modes + 1, dtype=float)
    return (n**2) * (np.pi**2) / (2.0 * width**2)


def infinite_square_well_wavefunction(x: np.ndarray, x_min: float, x_max: float, n: int) -> np.ndarray:
    width = x_max - x_min
    psi = np.sqrt(2.0 / width) * np.sin(n * np.pi * (x - x_min) / width)
    psi[(x < x_min) | (x > x_max)] = 0.0
    psi[[0, -1]] = 0.0
    return psi


def harmonic_oscillator_energies(k: float, offset: float, n_modes: int) -> np.ndarray:
    omega = np.sqrt(k)
    n = np.arange(0, n_modes, dtype=float)
    return omega * (n + 0.5) + offset


def harmonic_oscillator_wavefunction(x: np.ndarray, center: float, k: float, n: int) -> np.ndarray:
    omega = np.sqrt(k)
    xi = np.sqrt(omega) * (x - center)
    prefactor = (omega / np.pi) ** 0.25 / np.sqrt((2.0**n) * factorial(n))
    return prefactor * np.exp(-(xi**2) / 2.0) * eval_hermite(n, xi)


def build_benchmark_comparison(
    definition: PotentialDefinition,
    x: np.ndarray,
    states: list[Eigenstate],
) -> BenchmarkComparison | None:
    if definition.template_id == "harmonic_oscillator":
        k = float(definition.parameters["k"])
        offset = float(definition.parameters["offset"])
        center = float(definition.parameters["center"])
        energies = harmonic_oscillator_energies(k, offset, len(states))
        analytic_wavefunctions = {
            state.label: harmonic_oscillator_wavefunction(x, center, k, state.index - 1)
            for state in states
        }
        return BenchmarkComparison(
            benchmark_name="Harmonic Oscillator",
            rows=_build_rows(states, energies),
            analytic_wavefunctions=analytic_wavefunctions,
        )
    if definition.template_id == "infinite_square_well":
        width = float(definition.parameters["width"])
        center = float(definition.parameters["center"])
        x_min = center - (width / 2.0)
        x_max = center + (width / 2.0)
        energies = infinite_square_well_energies(width, len(states))
        analytic_wavefunctions = {
            state.label: infinite_square_well_wavefunction(x, x_min, x_max, state.index)
            for state in states
        }
        return BenchmarkComparison(
            benchmark_name="Infinite Square Well",
            rows=_build_rows(states, energies),
            analytic_wavefunctions=analytic_wavefunctions,
        )
    return None


def _build_rows(states: list[Eigenstate], analytic_energies: np.ndarray) -> list[BenchmarkRow]:
    return [
        BenchmarkRow(
            label=state.label,
            numerical_energy=state.energy,
            analytic_energy=float(analytic_energies[index]),
            energy_delta=float(state.energy - analytic_energies[index]),
        )
        for index, state in enumerate(states)
    ]

