import numpy as np

from schrodinger_solver_web.solver.potential_templates import (
    InputMode,
    PotentialDefinition,
    Segment,
    TEMPLATES,
    default_potential_definition,
    sample_potential,
)


def test_required_templates_exist():
    assert {"infinite_square_well", "harmonic_oscillator", "finite_square_well", "barrier"} <= set(TEMPLATES)


def test_template_editing_changes_sampled_potential():
    definition = default_potential_definition("harmonic_oscillator")
    definition.parameters["k"] = 2.0
    x = np.array([-1.0, 0.0, 1.0])
    sampled = sample_potential(definition, x)
    assert np.allclose(sampled, np.array([1.0, 0.0, 1.0]))


def test_piecewise_and_template_share_common_model():
    piecewise = PotentialDefinition(
        mode=InputMode.PIECEWISE,
        label="Piecewise",
        segments=[Segment(-1.0, 0.0, 1.0), Segment(0.0, 1.0, 2.0)],
    )
    template = default_potential_definition("barrier")
    assert isinstance(piecewise, PotentialDefinition)
    assert isinstance(template, PotentialDefinition)

