from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from schrodinger_solver_web.solver.potential_parser import PotentialExpressionError, evaluate_potential_expression, parse_potential_expression


class InputMode(StrEnum):
    TEMPLATE = "template"
    ANALYTIC = "analytic"
    PIECEWISE = "piecewise"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    value: float


@dataclass(frozen=True)
class TemplateParameter:
    key: str
    label: str
    default: float
    minimum: float | None = None
    maximum: float | None = None
    step: float = 0.1
    help_text: str = ""


@dataclass(frozen=True)
class PotentialTemplate:
    template_id: str
    label: str
    description: str
    parameters: tuple[TemplateParameter, ...]


@dataclass
class PotentialDefinition:
    mode: InputMode
    label: str
    template_id: str | None = None
    expression: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)


TEMPLATES: dict[str, PotentialTemplate] = {
    "infinite_square_well": PotentialTemplate(
        template_id="infinite_square_well",
        label="Infinite Square Well",
        description="Zero inside the well, very large walls outside. Benchmark comparison is available when the well spans the domain.",
        parameters=(
            TemplateParameter("center", "Center", 0.0, step=0.1),
            TemplateParameter("width", "Well Width", 1.0, minimum=0.1, step=0.1),
            TemplateParameter("wall_height", "Wall Height", 1e6, minimum=1e3, step=1e3),
        ),
    ),
    "harmonic_oscillator": PotentialTemplate(
        template_id="harmonic_oscillator",
        label="Harmonic Oscillator",
        description="Quadratic confining potential with benchmark comparison support.",
        parameters=(
            TemplateParameter("center", "Center", 0.0, step=0.1),
            TemplateParameter("k", "Spring Constant", 1.0, minimum=0.1, step=0.1),
            TemplateParameter("offset", "Energy Offset", 0.0, step=0.1),
        ),
    ),
    "finite_square_well": PotentialTemplate(
        template_id="finite_square_well",
        label="Finite Square Well",
        description="Constant-depth well on a finite domain.",
        parameters=(
            TemplateParameter("center", "Center", 0.0, step=0.1),
            TemplateParameter("width", "Well Width", 1.0, minimum=0.1, step=0.1),
            TemplateParameter("depth", "Well Depth", 12.0, minimum=0.1, step=0.5),
            TemplateParameter("outside", "Outside Value", 0.0, step=0.1),
        ),
    ),
    "barrier": PotentialTemplate(
        template_id="barrier",
        label="Barrier / Step",
        description="Piecewise constant barrier example.",
        parameters=(
            TemplateParameter("barrier_start", "Barrier Start", -0.3, step=0.1),
            TemplateParameter("barrier_end", "Barrier End", 0.3, step=0.1),
            TemplateParameter("barrier_height", "Barrier Height", 10.0, step=0.5),
            TemplateParameter("baseline", "Baseline", 0.0, step=0.1),
        ),
    ),
}


def default_potential_definition(template_id: str = "harmonic_oscillator") -> PotentialDefinition:
    template = TEMPLATES[template_id]
    return PotentialDefinition(
        mode=InputMode.TEMPLATE,
        label=template.label,
        template_id=template.template_id,
        parameters={param.key: param.default for param in template.parameters},
    )


def validate_segments(segments: list[Segment], x_min: float, x_max: float) -> list[str]:
    errors: list[str] = []
    if not segments:
        return ["At least one piecewise segment is required."]
    previous_end = x_min
    for index, segment in enumerate(segments, start=1):
        if segment.end <= segment.start:
            errors.append(f"Segment {index} must have end greater than start.")
        if abs(segment.start - previous_end) > 1e-9:
            errors.append("Piecewise segments must be contiguous and non-overlapping.")
            break
        previous_end = segment.end
    if abs(segments[0].start - x_min) > 1e-9 or abs(segments[-1].end - x_max) > 1e-9:
        errors.append("Piecewise segments must exactly span the selected domain.")
    return errors


def validate_potential_definition(definition: PotentialDefinition, x_min: float, x_max: float) -> list[str]:
    errors: list[str] = []
    if definition.mode == InputMode.ANALYTIC:
        if not definition.expression:
            errors.append("Analytic mode requires a potential expression.")
        else:
            try:
                parse_potential_expression(definition.expression)
            except PotentialExpressionError as exc:
                errors.append(str(exc))
    if definition.mode == InputMode.PIECEWISE:
        errors.extend(validate_segments(definition.segments, x_min, x_max))
    if definition.mode == InputMode.TEMPLATE:
        template_id = definition.template_id
        if template_id not in TEMPLATES:
            errors.append("A valid template must be selected.")
        parameters = definition.parameters
        if template_id in {"infinite_square_well", "finite_square_well"} and parameters.get("width", 0.0) <= 0.0:
            errors.append("Template width must be greater than zero.")
        if template_id == "harmonic_oscillator" and parameters.get("k", 0.0) <= 0.0:
            errors.append("Harmonic oscillator spring constant must be greater than zero.")
        if template_id == "barrier" and parameters.get("barrier_end", 0.0) <= parameters.get("barrier_start", 0.0):
            errors.append("Barrier end must be greater than barrier start.")
    return errors


def sample_potential(definition: PotentialDefinition, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if definition.mode == InputMode.ANALYTIC:
        if not definition.expression:
            raise ValueError("Analytic mode requires an expression.")
        return np.asarray(evaluate_potential_expression(definition.expression, x), dtype=float)
    if definition.mode == InputMode.PIECEWISE:
        return sample_piecewise_potential(definition.segments, x)
    if definition.mode == InputMode.TEMPLATE:
        return sample_template_potential(definition, x)
    raise ValueError(f"Unsupported potential mode '{definition.mode}'.")


def sample_piecewise_potential(segments: list[Segment], x: np.ndarray) -> np.ndarray:
    potential = np.full_like(x, np.nan, dtype=float)
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        mask = (x >= segment.start) & (x <= segment.end if is_last else x < segment.end)
        potential[mask] = segment.value
    if np.isnan(potential).any():
        raise ValueError("Piecewise segments did not cover the full domain.")
    return potential


def sample_template_potential(definition: PotentialDefinition, x: np.ndarray) -> np.ndarray:
    parameters = {key: float(value) for key, value in definition.parameters.items()}
    template_id = definition.template_id
    if template_id == "infinite_square_well":
        center = parameters["center"]
        width = parameters["width"]
        wall_height = parameters["wall_height"]
        inside = np.abs(x - center) <= (width / 2.0)
        return np.where(inside, 0.0, wall_height)
    if template_id == "harmonic_oscillator":
        center = parameters["center"]
        k = parameters["k"]
        offset = parameters["offset"]
        return 0.5 * k * (x - center) ** 2 + offset
    if template_id == "finite_square_well":
        center = parameters["center"]
        width = parameters["width"]
        depth = parameters["depth"]
        outside = parameters["outside"]
        inside = np.abs(x - center) <= (width / 2.0)
        return np.where(inside, outside - depth, outside)
    if template_id == "barrier":
        baseline = parameters["baseline"]
        barrier_start = parameters["barrier_start"]
        barrier_end = parameters["barrier_end"]
        barrier_height = parameters["barrier_height"]
        return np.where((x >= barrier_start) & (x <= barrier_end), barrier_height, baseline)
    raise ValueError(f"Unknown potential template '{template_id}'.")


def analytic_comparison_supported(definition: PotentialDefinition, x_min: float, x_max: float) -> bool:
    if definition.mode != InputMode.TEMPLATE or definition.template_id is None:
        return False
    if definition.template_id == "harmonic_oscillator":
        return True
    if definition.template_id == "infinite_square_well":
        center = float(definition.parameters["center"])
        width = float(definition.parameters["width"])
        return abs((center - (width / 2.0)) - x_min) < 1e-8 and abs((center + (width / 2.0)) - x_max) < 1e-8
    return False
