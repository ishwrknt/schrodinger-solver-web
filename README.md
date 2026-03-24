# 1D Schrodinger Solver Web App

This side project is a standalone Python web app for exploring numerical solutions of the 1D time-independent Schrodinger equation in dimensionless units with `hbar = 1` and `m = 1`.

## Scope

- 1D stationary finite-difference solver
- Dirichlet boundary conditions only
- Analytic, preset, and piecewise potential input modes
- Plotly visualizations for potentials, energies, and eigenfunctions
- CSV and PNG export for validated runs
- Benchmark comparisons for infinite square well and harmonic oscillator

This v1 is correctness-first. It rejects unsafe input, validates domain and grid settings before solve, and marks low-confidence states as rejected when post-solve checks fail.

## Setup

```bash
cd side-projects/schrodinger-solver-web
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run

The documented app command is:

```bash
schrodinger-solver-web
```

Fallback direct Streamlit command:

```bash
streamlit run src/schrodinger_solver_web/app.py
```

## Test

```bash
pytest
```

## Limitations

- No time-dependent solving
- No alternative boundary conditions
- No persistence for edited presets
- No continuum or scattering-state analysis
- No cloud or multi-user features

