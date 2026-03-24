from __future__ import annotations

import plotly.graph_objects as go

from schrodinger_solver_web.solver.benchmarks import BenchmarkComparison
from schrodinger_solver_web.solver.eigensolver import Eigenstate
from schrodinger_solver_web.solver.validation import StateValidation


def build_solution_figure(
    x,
    potential,
    states: list[Eigenstate],
    validations: list[StateValidation],
    title: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=potential, mode="lines", name="V(x)", line={"color": "#222222"}))
    validation_by_label = {validation.label: validation for validation in validations}
    for state in states:
        validation = validation_by_label[state.label]
        color = "#0b7285" if validation.accepted else "#c92a2a"
        dash = "solid" if validation.accepted else "dash"
        figure.add_trace(
            go.Scatter(
                x=x,
                y=state.psi + state.energy,
                mode="lines",
                name=f"{state.label} @ {state.energy:.4f}",
                line={"color": color, "dash": dash},
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="Energy / shifted eigenfunction",
        legend_title="States",
        template="plotly_white",
    )
    return figure


def build_energy_figure(states: list[Eigenstate], validations: list[StateValidation], title: str) -> go.Figure:
    figure = go.Figure()
    validation_by_label = {validation.label: validation for validation in validations}
    for state in states:
        validation = validation_by_label[state.label]
        color = "#2b8a3e" if validation.accepted else "#e03131"
        figure.add_hline(y=state.energy, line_color=color, annotation_text=state.label)
    figure.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis_title="Energy",
        template="plotly_white",
        showlegend=False,
    )
    return figure


def build_comparison_figure(
    x,
    states: list[Eigenstate],
    comparison: BenchmarkComparison,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    for state in states:
        figure.add_trace(go.Scatter(x=x, y=state.psi, mode="lines", name=f"{state.label} numerical"))
        figure.add_trace(
            go.Scatter(
                x=x,
                y=comparison.analytic_wavefunctions[state.label],
                mode="lines",
                name=f"{state.label} analytic",
                line={"dash": "dash"},
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="psi(x)",
        template="plotly_white",
    )
    return figure

