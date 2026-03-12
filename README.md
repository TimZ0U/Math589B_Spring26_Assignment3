# Square Membrane LQR Project

A GitHub startup repository for a project on **LQR stabilization of a
vibrating square membrane** with a single localized actuator.

The mathematical heart is a controlled wave equation on the unit square,

$$
u_{tt} = c^2 \Delta u + b(t) \psi(x,y), \quad (x,y) \in (0,1)^2,
$$

with fixed boundary conditions \( u = 0 \) on the boundary. After expanding in
sine eigenfunctions and truncating to finitely many modes, the PDE becomes a
linear state-space system

$$
\dot{x} = A x + B b,
$$

for which one can design an infinite-horizon LQR feedback

$$
b(t) = -K x(t).
$$

This repository is set up so students can move from theory to code without
having to build the whole scaffolding from scratch.

## Repository layout

- `handout/handout.tex` - project handout in LaTeX
- `handout/handout.pdf` - compiled PDF handout
- `src/python/modal_lqr.py` - Python starter library plus demo driver
- `src/python/run_demo.py` - generate plots and optional animation frames
- `src/python/scan_actuator.py` - compare actuator locations
- `tests/test_coupling.py` - basic coupling tests
- `tests/test_reconstruction.py` - basic reconstruction tests
- `PROJECT_TASKS.md` - student-facing milestones and suggested checkpoints
- `requirements.txt` - Python dependencies

## Mathematical highlights

For the unit square with Dirichlet boundary conditions, the Laplacian
eigenfunctions are

$$
\phi_{mn}(x,y) = 2 \sin(m \pi x) \sin(n \pi y),
$$

with eigenvalues

$$
\lambda_{mn} = \pi^2 (m^2 + n^2).
$$

If the actuator is idealized as a point actuator at \( (x_0,y_0) \), then the
modal coupling coefficient is

$$
\beta_{mn} = \phi_{mn}(x_0,y_0) = 2 \sin(m \pi x_0) \sin(n \pi y_0).
$$

A centered actuator at \( (1/2, 1/2) \) misses every mode with even \( m \) 
or even \( n \), so this repo uses an **off-center** default actuator at
\( (0.37, 0.61) \).

## Suggested student workflow

1. Read the handout and derive the modal equations.
2. Build \( A \), \( B \), and the LQR gain \( K \).
3. Simulate a closed-loop response for a chosen initial condition.
4. Reconstruct the membrane shape on a grid.
5. Compare at least two actuator locations.
6. Discuss which modes are poorly actuated or completely missed.

The file `PROJECT_TASKS.md` turns this into a week-by-week sequence.

## Python quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.python.run_demo
```

This will create an `outputs/` directory containing:

- energy-vs-time plot
- control-vs-time plot
- membrane snapshots
- an optional GIF animation if Pillow is available

To compare actuator locations,

```bash
python -m src.python.scan_actuator
```

## Suggested research questions

1. How does actuator location affect controllability of the truncated system?
2. What happens if the actuator is placed at the center?
3. How sensitive is the closed-loop performance to the weights in the LQR cost?
4. How many modes are needed before the qualitative behavior stabilizes?
5. How does a localized actuator patch compare with the ideal point-actuator
   model?

## Notes for the instructor

This repo deliberately sits in a sweet spot between PDEs and control:

- the PDE model is genuine,
- the eigenfunctions are explicit,
- the numerical implementation is manageable,
- symmetry obstructions show up clearly.

The project can be assigned at different levels:

- **intro control / numerical methods**: derive and simulate the truncated
  system,
- **advanced PDE/control**: discuss stabilizability and symmetry,
- **computational project**: compare point actuation and patch actuation,
  animate the membrane, and scan actuator locations.
