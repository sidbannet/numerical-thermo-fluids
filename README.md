# numerical-thermo-fluids

Numerical methods, solvers, and simulation framework for thermodynamics, multiphase gas/liquid flow dynamics, and compressible flow analysis. This package provides robust 1D Computational Fluid Dynamics (CFD) solvers written entirely in Python, heavily relying on true `numpy` C-level broadcasting for optimal speed.

## Key Features

- **Robust Solvers**: Includes both standard Finite-Volume (MacCormack) and High-Resolution Total Variation Diminishing (TVD) schemes utilizing Roe-averaged flux vector splitting.
- **Real-Fluid Property Integration**: Seamlessly interfaces with [CoolProp](http://www.coolprop.org/) to handle complex working fluids (water, refrigerants, cryogens) spanning liquid, sub-critical, and super-critical vapor states.
- **Multiphase Speed of Sound**: Includes rigorous Homogeneous Equilibrium Mixture (HEM) fallback calculations in the two-phase region to determine the effective mixture speed of sound via isentropic perturbations ($dv/dP)_s$.
- **Gravity & Mass Injection**: Supports momentum and energy source terms, modeling physical mass injections via generalized compressible orifice equations and elevational gravitational forces.
- **Extreme Vectorization**: 100% vectorized array-math pipelines. Python-level iteration has been eliminated in flux, state-property, and boundary condition evaluation using integer logic switching, squeezing maximum performance from the Python interpreter.
- **PEP 8 Compliant**: High-quality docstrings and standard coding styles enforced by `flake8`.

## Requirements
- Python >= 3.8
- `numpy`
- `scipy`
- `CoolProp` (for real fluid properties)

## Installation
Clone the repository and install the dependencies natively using `pip`:

```bash
git clone https://github.com/yourusername/numerical-thermo-fluids.git
cd numerical-thermo-fluids
pip install -r requirements.txt
```

## Basic Usage

You can build a geometrical pipe model and pass it into a `FlowSolver` or a `TVDSolver` for high-resolution shock capture.

```python
import numpy as np
from cfdlite.onedim import PipeModel, TVDSolver, BoundaryType, BoundaryCategory
from cfdlite.eos import CoolPropEOS

# 1. Choose Real-Fluid Equation of State
eos = CoolPropEOS(fluid='Water')

# 2. Build 1D Pipeline Model
x = np.linspace(0, 10, 100) # 10 meters, 100 cells
area = np.ones_like(x) * 0.05 # 0.05 m^2 cross section
elev = np.zeros_like(x) # 0 degree elevation (flat)

model = PipeModel(x, area, eos=eos, elev_angle=elev)

# 3. Initial States
rho_init = np.ones_like(x) * 1.2
v_init = np.zeros_like(x)
e_init = np.ones_like(x) * 2.5e5

uu0 = np.array([rho_init, rho_init * v_init, rho_init * (e_init + 0.5 * v_init**2)])

# 4. Create TVD Solver
solver = TVDSolver(
    model=model,
    initial_solution=uu0,
    intake_boundary={
        BoundaryCategory.MF: {BoundaryType.DI: 1.0}, # Dirichlet mass flow
        BoundaryCategory.TP: {BoundaryType.NE: 0.0},
        BoundaryCategory.TH: {BoundaryType.DI: 300000.0}
    }
)

# 5. Advance Solution in Time
solver.update(dt=1e-5)
print(solver.uu)
```

## Testing
The `tests/` directory contains standard unit-tests covering boundary application, EOS fidelity, array propagation, two-phase logic, and gravity terms.

To run tests:
```bash
python3 -m unittest discover tests
```
