from __future__ import annotations

import sys
from pathlib import Path

# Add the 'src' directory to sys.path for Streamlit Cloud
src_path = str(Path(__file__).parents[2])
if src_path not in sys.path:
    sys.path.append(src_path)

from dataclasses import dataclass

import streamlit as st

from schrodinger_solver_web.export.csv_export import export_csv
from schrodinger_solver_web.export.png_export import export_plot_png
from schrodinger_solver_web.solver.benchmarks import build_benchmark_comparison
from schrodinger_solver_web.solver.eigensolver import Eigenstate, solve_eigenstates
from schrodinger_solver_web.solver.grid import GridSpec, build_grid
from schrodinger_solver_web.solver.hamiltonian import assemble_hamiltonian
from schrodinger_solver_web.solver.potential_templates import (
    PotentialDefinition,
    analytic_comparison_supported,
    sample_potential,
)
from schrodinger_solver_web.solver.validation import StateValidation, validate_postsolve, validate_presolve
from schrodinger_solver_web.ui.plots import build_comparison_figure, build_energy_figure, build_solution_figure
from schrodinger_solver_web.ui.sidebar import SidebarConfig, render_sidebar


@dataclass(frozen=True)
class SolveResult:
    grid_spec: GridSpec
    potential_definition: PotentialDefinition
    x: object
    potential: object
    states: list[Eigenstate]
    validations: list[StateValidation]


def render_app() -> None:
    st.set_page_config(page_title="1D Schrodinger Solver", layout="wide")
    st.title("1D Schrodinger Solver")
    st.caption("Dimensionless stationary solver with hbar = 1 and m = 1.")

    config = render_sidebar()
    errors = validate_presolve(config.grid_spec, config.potential_definition)
    if errors:
        for error in errors:
            st.error(error)
    else:
        st.success("Pre-solve validation passed.")

    if config.run_solver and not errors:
        result = _run_solver(config)
        st.session_state["solve_result"] = result

    result = st.session_state.get("solve_result")
    if isinstance(result, SolveResult):
        _render_result(result, config.compare_supported_benchmarks)
    else:
        st.info("Configure a potential, choose the grid, and run the solver.")


def _run_solver(config: SidebarConfig) -> SolveResult:
    grid = build_grid(config.grid_spec)
    potential = sample_potential(config.potential_definition, grid.x)
    hamiltonian = assemble_hamiltonian(grid, potential)
    states = solve_eigenstates(hamiltonian, config.grid_spec.n_modes)
    validations = validate_postsolve(config.grid_spec, config.potential_definition, states)
    return SolveResult(
        grid_spec=config.grid_spec,
        potential_definition=config.potential_definition,
        x=grid.x,
        potential=potential,
        states=states,
        validations=validations,
    )


def _render_result(result: SolveResult, compare_supported_benchmarks: bool) -> None:
    accepted_count = sum(validation.accepted for validation in result.validations)
    rejected_count = len(result.validations) - accepted_count
    st.subheader("Solve Summary")
    st.write(f"Accepted states: {accepted_count} | Rejected states: {rejected_count}")

    title = (
        f"{result.potential_definition.label} | domain=[{result.grid_spec.x_min}, {result.grid_spec.x_max}] "
        f"| points={result.grid_spec.num_points} | modes={result.grid_spec.n_modes}"
    )
    solution_figure = build_solution_figure(
        result.x,
        result.potential,
        result.states,
        result.validations,
        title=title,
    )
    energy_figure = build_energy_figure(result.states, result.validations, title="Energy Levels")

    left, right = st.columns([2, 1])
    left.plotly_chart(solution_figure, use_container_width=True)
    right.plotly_chart(energy_figure, use_container_width=True)
    st.caption("All computed states are plotted. Accepted states use solid teal/green styling; rejected diagnostic states use dashed red styling.")

    if rejected_count == len(result.validations):
        st.warning("All requested states were classified as diagnostic for this configuration. The plots above still show the numerical solution candidates.")

    with st.expander("Validation details", expanded=rejected_count > 0):
        for validation in result.validations:
            if validation.accepted:
                st.success(
                    f"{validation.label}: accepted. Drift={validation.eigenvalue_drift:.2e}, "
                    f"edge amplitudes=({validation.boundary_left:.2e}, {validation.boundary_right:.2e})"
                )
            else:
                st.warning(f"{validation.label}: rejected. {' '.join(validation.reasons)}")

    comparison_supported = analytic_comparison_supported(
        result.potential_definition,
        result.grid_spec.x_min,
        result.grid_spec.x_max,
    )
    if compare_supported_benchmarks and comparison_supported:
        comparison = build_benchmark_comparison(result.potential_definition, result.x, result.states)
        if comparison is not None:
            st.subheader(f"Benchmark Comparison: {comparison.benchmark_name}")
            st.dataframe(
                [
                    {
                        "state": row.label,
                        "numerical": row.numerical_energy,
                        "analytic": row.analytic_energy,
                        "delta": row.energy_delta,
                    }
                    for row in comparison.rows
                ],
                use_container_width=True,
            )
            st.plotly_chart(
                build_comparison_figure(
                    result.x,
                    result.states,
                    comparison,
                    title=f"{comparison.benchmark_name} comparison",
                ),
                use_container_width=True,
            )
    elif compare_supported_benchmarks:
        st.info("Analytic comparison is unavailable for the current potential configuration.")

    accepted_states = [state for state, validation in zip(result.states, result.validations, strict=True) if validation.accepted]
    accepted_validations = [validation for validation in result.validations if validation.accepted]
    if accepted_states:
        csv_data = export_csv(
            grid_spec=result.grid_spec,
            definition=result.potential_definition,
            x=result.x,
            potential=result.potential,
            states=accepted_states,
            validations=accepted_validations,
        )
        st.download_button("Export CSV", data=csv_data, file_name="schrodinger_solution.csv", mime="text/csv")
        try:
            png_bytes = export_plot_png(solution_figure)
        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"PNG export is unavailable: {exc}")
        else:
            st.download_button(
                "Export PNG",
                data=png_bytes,
                file_name="schrodinger_solution.png",
                mime="image/png",
            )
    else:
        st.info("Exports are enabled once at least one state passes validation.")


def main() -> None:
    if st.runtime.exists():
        render_app()
        return
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
