import plotly.graph_objects as go

from schrodinger_solver_web.export.csv_export import export_csv
from schrodinger_solver_web.export.png_export import export_plot_png
from schrodinger_solver_web.solver.eigensolver import solve_eigenstates
from schrodinger_solver_web.solver.grid import GridSpec, build_grid
from schrodinger_solver_web.solver.hamiltonian import assemble_hamiltonian
from schrodinger_solver_web.solver.potential_templates import default_potential_definition, sample_potential
from schrodinger_solver_web.solver.validation import validate_postsolve


def test_csv_export_contains_reproducibility_metadata():
    spec = GridSpec(x_min=-6.0, x_max=6.0, num_points=401, n_modes=2)
    definition = default_potential_definition("harmonic_oscillator")
    grid = build_grid(spec)
    potential = sample_potential(definition, grid.x)
    states = solve_eigenstates(assemble_hamiltonian(grid, potential), 2)
    validations = validate_postsolve(spec, definition, states)
    output = export_csv(spec, definition, grid.x, potential, states, validations)
    assert "# input_mode" in output
    assert "# grid_points" in output
    assert "n=1_psi" in output


def test_png_export_returns_png_bytes():
    figure = go.Figure()
    figure.update_layout(title="Barrier | points=401")
    image = export_plot_png(figure)
    assert image.startswith(b"\x89PNG")
