from __future__ import annotations

from schrodinger_solver_web.solver.potential_templates import PotentialTemplate


def template_summary(template: PotentialTemplate) -> str:
    return f"{template.label}: {template.description}"

