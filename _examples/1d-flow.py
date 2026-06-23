## Import packages and modules
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from cfdlite.onedim import PipeModel as Model
from cfdlite.onedim import FlowSolver as Solver
from cfdlite.onedim import BoundaryType as Type
from cfdlite.onedim import BoundaryCategory as Category
## Setup model
x_mesh = np.linspace(0, 1, 1000)
area = np.full(x_mesh.shape, np.pi * 0.1 ** 2 / 4)
dA = (np.cos(2 * np.pi * x_mesh) - 1) / 2 * area[0] / 2
area += dA
gamma = 1.4
mdl = Model(
    x=x_mesh,
    area=area,
    perimeter=np.full_like(x_mesh, 1),
    mdot_w=np.full_like(x_mesh, 0),
    tau_w=np.full_like(x_mesh, 0),
    q_w=np.full_like(x_mesh, 0),
)
## Specify initial condition
uu_initial = np.array(
    [
        np.full_like(x_mesh, 1.0) * area,
        np.full_like(x_mesh, 0.0) * area,
        np.full_like(x_mesh, 250000) * area,
    ]
)
flux = mdl.flux(uu=uu_initial,
                gamma=np.full_like(x_mesh, gamma))
thermo = mdl.thermo(uu=uu_initial,
                    gamma=np.full_like(x_mesh, gamma))
source = mdl.source(uu=uu_initial,
                    gamma=np.full_like(x_mesh, gamma))
## Specify boundary conditions
intake = {
    Category.MF: {Type.NE: float(0)},
    Category.TP: {Type.DI: float(120000)},
    Category.TH: {Type.NE: float(0)},
}
outlet = {
    Category.MF: {Type.NE: float(0)},
    Category.TP: {Type.DI: float(100000)},
    Category.TH: {Type.NE: float(0)},
}
soln = Solver(model=mdl, initial_solution=uu_initial,
              intake_boundary=intake, outlet_boundary=outlet)
soln.gamma = np.full_like(x_mesh, gamma)
##
soln.update()
mdl.thermo(uu=soln.uu, gamma=soln.gamma)
## Time marching
for i in range(int(2500)).__iter__():
    soln.update()
## Plot the final state
fig = plt.figure('Fig: Solution at final time')
axs = fig.subplots(nrows=4, ncols=1, sharex=True)
axs[0].plot(
    x_mesh, mdl.thermo(uu=soln.uu, gamma=soln.gamma)['velocity'],
)
axs[0].set_ylabel('[m/s]')
axs[0].set_title('Velocity')
axs[1].plot(
    x_mesh, mdl.thermo(uu=soln.uu, gamma=soln.gamma)['pressure'] / 1e5,
)
axs[1].set_ylabel('[bar]')
axs[1].set_title('Pressure')
axs[2].plot(
    x_mesh, mdl.thermo(uu=soln.uu, gamma=soln.gamma)['density'],
)
axs[2].set_ylabel('[kg/s]')
axs[2].set_title('Density')
axs[3].plot(
    x_mesh,
    mdl.thermo(uu=soln.uu, gamma=soln.gamma)['internal_energy'] / 714.28,
)
axs[3].set_ylabel('[K]')
axs[3].set_title('Temperature')
axs[3].set_xlabel('x [m]')
_ = [ax.grid(True) for ax in axs.flat]
##
