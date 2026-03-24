from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from schrodinger_solver_web.solver.grid import GridSpec
from schrodinger_solver_web.solver.potential_templates import (
    InputMode,
    PotentialDefinition,
    Segment,
    TEMPLATES,
    default_potential_definition,
)
from schrodinger_solver_web.ui.templates import template_summary


@dataclass(frozen=True)
class SidebarConfig:
    grid_spec: GridSpec
    potential_definition: PotentialDefinition
    compare_supported_benchmarks: bool
    run_solver: bool


def render_sidebar() -> SidebarConfig:
    st.sidebar.header("Configuration")

    x_min = st.sidebar.number_input("x min", value=-5.0, step=0.5)
    x_max = st.sidebar.number_input("x max", value=5.0, step=0.5)
    num_points = int(st.sidebar.number_input("Grid points", min_value=51, max_value=5001, value=401, step=50))
    n_modes = int(st.sidebar.number_input("Modes", min_value=1, max_value=12, value=4, step=1))
    compare_supported_benchmarks = st.sidebar.toggle("Show analytic comparison", value=True)

    mode = InputMode(
        st.sidebar.radio(
            "Potential mode",
            options=[mode.value for mode in InputMode],
            format_func=lambda value: value.capitalize(),
        )
    )
    if mode == InputMode.TEMPLATE:
        potential_definition = _render_template_editor()
    elif mode == InputMode.ANALYTIC:
        expression = st.sidebar.text_input("V(x)", value="0.5 * x**2")
        potential_definition = PotentialDefinition(
            mode=InputMode.ANALYTIC,
            label="Analytic Potential",
            expression=expression,
        )
    else:
        potential_definition = _render_piecewise_editor(x_min=x_min, x_max=x_max)

    run_solver = st.sidebar.button("Solve", type="primary")
    return SidebarConfig(
        grid_spec=GridSpec(x_min=x_min, x_max=x_max, num_points=num_points, n_modes=n_modes),
        potential_definition=potential_definition,
        compare_supported_benchmarks=compare_supported_benchmarks,
        run_solver=run_solver,
    )


def _render_template_editor() -> PotentialDefinition:
    template_id = st.sidebar.selectbox(
        "Template",
        options=list(TEMPLATES.keys()),
        format_func=lambda value: TEMPLATES[value].label,
    )
    template = TEMPLATES[template_id]
    st.sidebar.caption(template_summary(template))
    base_definition = default_potential_definition(template_id)
    parameters: dict[str, float] = {}
    for parameter in template.parameters:
        parameters[parameter.key] = float(
            st.sidebar.number_input(
                parameter.label,
                value=float(base_definition.parameters[parameter.key]),
                min_value=parameter.minimum,
                max_value=parameter.maximum,
                step=parameter.step,
                help=parameter.help_text or None,
                key=f"{template_id}_{parameter.key}",
            )
        )
    return PotentialDefinition(
        mode=InputMode.TEMPLATE,
        label=template.label,
        template_id=template_id,
        parameters=parameters,
    )


def _render_piecewise_editor(x_min: float, x_max: float) -> PotentialDefinition:
    segment_count = int(st.sidebar.number_input("Segments", min_value=1, max_value=6, value=3, step=1))
    segments: list[Segment] = []
    current_start = x_min
    st.sidebar.caption("Segments must remain contiguous across the selected domain.")
    for index in range(segment_count):
        st.sidebar.markdown(f"Segment {index + 1}")
        if index == segment_count - 1:
            end_value = x_max
        else:
            end_value = float(
                st.sidebar.number_input(
                    f"End {index + 1}",
                    min_value=current_start + 1e-6,
                    max_value=x_max,
                    value=current_start + ((x_max - x_min) / segment_count),
                    step=0.1,
                    key=f"piecewise_end_{index}",
                )
            )
        potential_value = float(
            st.sidebar.number_input(
                f"V {index + 1}",
                value=0.0 if index != 1 else 10.0,
                step=0.5,
                key=f"piecewise_value_{index}",
            )
        )
        segments.append(Segment(start=current_start, end=end_value, value=potential_value))
        current_start = end_value
    return PotentialDefinition(
        mode=InputMode.PIECEWISE,
        label="Piecewise Potential",
        segments=segments,
    )

