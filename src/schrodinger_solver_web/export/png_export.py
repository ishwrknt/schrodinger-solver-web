from __future__ import annotations

from plotly.graph_objects import Figure


def export_plot_png(figure: Figure) -> bytes:
    return figure.to_image(format="png")

