# `cfdlite` Theory Manual

## 1. Governing Equations
The physical model in `cfdlite` represents the quasi-one-dimensional compressible Euler equations with source terms to account for variable area, wall friction, and heat transfer.

The conservative form of the equations can be written as:

$$ \frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} = \mathbf{S} $$

where the state vector $\mathbf{U}$, flux vector $\mathbf{F}$, and source vector $\mathbf{S}$ are defined as:

$$
\mathbf{U} = \begin{bmatrix}
\rho \\
\rho u \\
\rho E
\end{bmatrix}, \quad
\mathbf{F} = \begin{bmatrix}
\rho u \\
\rho u^2 + p \\
u(\rho E + p)
\end{bmatrix}, \quad
\mathbf{S} = \begin{bmatrix}
S_{mass} \\
S_{mom} \\
S_{energy}
\end{bmatrix}
$$

Here:
- $\rho$ is the density.
- $u$ is the velocity.
- $E = e + \frac{1}{2}u^2$ is the total specific energy, where $e$ is the specific internal energy.
- $p$ is the pressure, given by the ideal gas equation of state: $p = (\gamma - 1)\rho e = (\gamma - 1)\left(\rho E - \frac{1}{2}\rho u^2\right)$.

The source terms in the `PipeModel` class account for:
- Mass addition/removal: $\dot{m}_w$
- Variable area $A(x)$: $\frac{p}{A} \frac{\partial A}{\partial x}$
- Wall friction: $\tau_w$ (wall shear stress)
- Heat transfer: $\dot{q}_w$

Specifically:
$$
\mathbf{S} = \begin{bmatrix}
\dot{m}_w P_w \\
\frac{p}{A} \frac{\partial A}{\partial x} - \tau_w P_w + \dot{m}_w u_{inj} \cos(\theta) P_w \\
\dot{q}_w P_w + \dot{m}_w H_{inj} P_w
\end{bmatrix}
$$
where $P_w$ is the wetted perimeter, $u_{inj}$ is the injection velocity, $\theta$ is the angle of injection relative to the pipe axis, and $H_{inj}$ is the stagnation enthalpy of the injected mass.

### 1.1 Dynamic Orifice Flow Injection

When a plenum (manifold) is used to inject gas into the pipe, the injection mass flow rate $\dot{m}_w$ and velocity $u_{inj}$ can be computed dynamically based on the local static pressure $p$ in the pipe.

Assuming a plenum with stagnation pressure $P_{0,inj}$ and stagnation temperature $T_{0,inj}$, the flow through the orifice of effective area $A_{inj}$ is governed by compressible isentropic relations. The critical pressure ratio is:
$$ \left(\frac{p}{P_{0,inj}}\right)_{crit} = \left(\frac{2}{\gamma+1}\right)^{\frac{\gamma}{\gamma-1}} $$

If $\frac{p}{P_{0,inj}} \le \left(\frac{p}{P_{0,inj}}\right)_{crit}$, the flow is **choked** (sonic) at the orifice:
- $M_{inj} = 1$

If $\frac{p}{P_{0,inj}} > \left(\frac{p}{P_{0,inj}}\right)_{crit}$, the flow is **unchoked** (subsonic):
- $M_{inj} = \sqrt{ \frac{2}{\gamma-1} \left[ \left(\frac{P_{0,inj}}{p}\right)^{\frac{\gamma-1}{\gamma}} - 1 \right] }$

The static temperature and velocity at the injection plane are then computed as:
$$ T_{inj} = \frac{T_{0,inj}}{1 + \frac{\gamma-1}{2} M_{inj}^2} $$
$$ u_{inj} = M_{inj} \sqrt{\gamma R T_{inj}} $$
$$ \rho_{inj} = \frac{P_{0,inj}}{R T_{0,inj}} \left(\frac{T_{inj}}{T_{0,inj}}\right)^{\frac{1}{\gamma-1}} $$
$$ \dot{m}_w = \rho_{inj} u_{inj} A_{inj} $$

Where $\gamma$ is the ratio of specific heats of the injected gas (which defaults to the main pipe gas if not specified).

---

## 2. Numerical Schemes

The library provides two primary flow solvers: a MacCormack finite-volume solver and a high-resolution TVD solver.

### 2.1 MacCormack Scheme (`FlowSolver`)

The `FlowSolver` implements a classic MacCormack predictor-corrector finite volume scheme. This is a two-step explicit scheme that is formally second-order accurate in both space and time for linear problems.

**Predictor Step (Forward difference):**
$$ \mathbf{U}_i^* = \mathbf{U}_i^n - \frac{\Delta t}{\Delta x} (\mathbf{F}_{i+1}^n - \mathbf{F}_i^n) + \Delta t \mathbf{S}_i^n $$

**Corrector Step (Backward difference):**
$$ \mathbf{U}_i^{**} = \mathbf{U}_i^* - \frac{\Delta t}{\Delta x} (\mathbf{F}_{i}^* - \mathbf{F}_{i-1}^*) + \Delta t \mathbf{S}_i^* $$

**Solution Update:**
$$ \mathbf{U}_i^{n+1} = \frac{1}{2} (\mathbf{U}_i^n + \mathbf{U}_i^{**}) $$

### 2.2 TVD Solver with MUSCL Reconstruction (`TVDSolver`)

The `TVDSolver` implements a second-order Total Variation Diminishing (TVD) scheme, which is much more robust at capturing shocks without non-physical numerical oscillations.

#### 2.2.1 MUSCL Reconstruction

To achieve higher-order spatial accuracy, the states are extrapolated from cell centers to the cell faces (interfaces) using the Monotone Upstream-centered Schemes for Conservation Laws (MUSCL) approach.

Let $\mathbf{U}_L$ and $\mathbf{U}_R$ be the reconstructed states at the left and right sides of the interface $i+1/2$:
$$ \mathbf{U}_{i+1/2}^L = \mathbf{U}_i + \frac{1}{2} \phi(r_i) (\mathbf{U}_i - \mathbf{U}_{i-1}) $$
$$ \mathbf{U}_{i+1/2}^R = \mathbf{U}_{i+1} - \frac{1}{2} \phi(r_{i+1}) (\mathbf{U}_{i+2} - \mathbf{U}_{i+1}) $$

where $\phi(r)$ is a flux limiter function designed to ensure the TVD property by dropping the scheme to first-order near discontinuities. The user can select from several limiters:
- **Minmod**
- **Van Leer**
- **Superbee**
- **Monotonized Central (MC)**

#### 2.2.2 Roe's Approximate Riemann Solver

The flux at the interface is computed using Roe's approximate Riemann solver. It relies on the Roe-averaged states (e.g., $\tilde{u}, \tilde{H}, \tilde{a}$) to compute the eigenvalues and eigenvectors of the flux Jacobian.

The Roe flux is given by:
$$ \mathbf{F}_{i+1/2}^{Roe} = \frac{1}{2}(\mathbf{F}(\mathbf{U}_L) + \mathbf{F}(\mathbf{U}_R)) - \frac{1}{2} \sum_{k=1}^3 |\tilde{\lambda}_k| \tilde{\alpha}_k \tilde{\mathbf{r}}_k $$

To prevent non-physical expansion shocks, a **Harten entropy fix** is applied to the eigenvalues $\tilde{\lambda}_k$, specifically for the acoustic waves (1 and 3). If the wave speed is smaller than a given threshold $\delta$, artificial dissipation is added:
$$ |\lambda| \leftarrow \frac{\lambda^2 + \delta^2}{2\delta} \quad \text{for } |\lambda| < \delta $$

#### 2.2.3 Time Integration (SSP-RK2)

Time integration is performed using the Strong Stability Preserving Runge-Kutta (SSP-RK2) method, also known as Heun's method. This maintains the TVD property of the spatial discretization over time.

1. **Stage 1:** $\mathbf{U}^* = \mathbf{U}^n + \Delta t \mathcal{L}(\mathbf{U}^n)$
2. **Stage 2:** $\mathbf{U}^{n+1} = \frac{1}{2}\mathbf{U}^n + \frac{1}{2}(\mathbf{U}^* + \Delta t \mathcal{L}(\mathbf{U}^*))$

where $\mathcal{L}(\mathbf{U}) = -\frac{\partial \mathbf{F}}{\partial x} + \mathbf{S}$.

---

## 3. Boundary Conditions

Both solvers handle boundary conditions by explicitly modifying the boundary fluxes to account for Dirichlet (specified value) and Neumann (specified gradient) conditions on:
- Mass Flow Rate
- Total Pressure
- Total Enthalpy
