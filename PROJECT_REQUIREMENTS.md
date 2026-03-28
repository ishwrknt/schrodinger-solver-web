# 1D Schrodinger Solver Web App: Project Requirements & Brainstorming

## 1. Project Overview
A standalone, correctness-first Python web application for exploring numerical solutions to the 1D time-independent Schrödinger equation. The app uses dimensionless units ($\hbar = 1$, $m = 1$) to focus on educational exploration and validation of quantum states.

## 2. Core Requirements (Current Scope)
- **Numerical Solver:**
  - 1D stationary finite-difference method.
  - Support for Dirichlet boundary conditions (wavefunction vanishes at boundaries).
- **Potential Definitions:**
  - Analytic input modes.
  - Predefined preset modes (e.g., Infinite Square Well, Harmonic Oscillator).
  - Piecewise potential input modes.
- **Validation & Correctness:**
  - Strict domain and grid setting validation prior to solving.
  - Post-solve state validation to mark low-confidence states as rejected.
- **Visualization:**
  - Interactive plotting using Plotly for potentials, energy levels, and eigenfunctions.
- **Export Capabilities:**
  - CSV export for raw numerical data of validated runs.
  - PNG export for plots of validated runs.
- **Benchmarking:**
  - Built-in analytic comparisons for known systems (Infinite Square Well, Harmonic Oscillator).

## 3. Technical Requirements
- **Language:** Python 3.11+
- **Core Libraries:** `numpy` (>=1.26), `scipy` (>=1.12)
- **Web Framework:** `streamlit` (>=1.42)
- **Visualization:** `plotly` (>=5.24), `kaleido` (>=0.2.1) for static image exports
- **Testing:** `pytest` (>=8.3)
- **Deployment:** Streamlit Community Cloud compatible via `streamlit_app.py` entry point.

## 4. Current Limitations
- Solves only the time-independent equation; no time-dependent evolution.
- Restricted to Dirichlet boundary conditions (no periodic or Neumann BCs).
- No analysis of continuum or scattering states (bound states only).
- No persistence/database for saving user-edited presets across sessions.
- Single-user architecture; no cloud collaboration features.

---

## 5. Brainstorming & Future Enhancements

*This section outlines potential future expansions for the project.*

### 5.1 Physics & Solvers
- **Time-Dependent Evolution:** Animate wavepacket dynamics over time using techniques like the Crank-Nicolson method or Split-Operator method.
- **Alternative Boundary Conditions:** Add support for Periodic Boundary Conditions (useful for crystal lattice models) and Neumann Boundary Conditions.
- **Scattering States:** Implement tools to calculate transmission and reflection coefficients for potential barriers (e.g., quantum tunneling).
- **2D/3D Solvers:** Extend the finite-difference solver to 2D grids (e.g., quantum corrals).

### 5.2 User Interface & Experience
- **Interactive Potential Drawing:** Allow users to draw arbitrary potential shapes directly on the canvas with their mouse.
- **Session State & Persistence:** Allow users to save their current configuration (potentials, grid settings) to a downloadable JSON file or local storage.
- **Advanced Visualizations:** Implement Wigner distribution plots or phase space visualizations.
- **Unit Conversions:** Add a toggle to switch from dimensionless units to physical units (eV, nm, etc.) for real-world material modeling.

### 5.3 Software Architecture
- **API Backend:** Decouple the solver logic into a REST/FastAPI backend to support programmatic access or alternative frontends (React/Vue).
- **Containerization:** Provide a `Dockerfile` for isolated, reproducible local execution.
- **Performance Optimization:** Explore JIT compilation (via Numba) for the core eigensolver matrix assembly to handle extremely dense grids.
