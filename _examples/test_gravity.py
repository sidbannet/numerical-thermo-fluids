import numpy as np
from cfdlite.onedim import PipeModel

def main():
    N = 10
    L = 10.0
    x = np.linspace(0, L, N)
    area = np.ones_like(x) * 0.1
    perimeter = np.ones_like(x) * 0.4
    
    # Incline the pipe vertically (90 degrees)
    elev_angle = np.ones_like(x) * (np.pi / 2.0)
    
    # Initialize PipeModel with gravity and elevation
    model = PipeModel(
        x=x, 
        area=area, 
        perimeter=perimeter,
        g_gravity=9.81,
        elev_angle=elev_angle
    )
    
    gamma_arr = np.ones_like(x) * 1.4
    
    # Set up static fluid column
    # Density = 1000 kg/m^3 (approx water, though using ideal gas logic here for structural testing)
    rho = np.ones_like(x) * 1000.0
    u = np.zeros_like(x)
    
    # To test hydrostatic pressure, we need the pressure gradient to balance gravity:
    # dp/dx = -rho * g * sin(theta)
    # If theta = pi/2, dp/dx = -rho * g = -1000 * 9.81 = -9810
    # Let's set pressure linearly decreasing
    p0 = 100000.0
    p = p0 - rho * 9.81 * x
    
    e = p / (rho * (1.4 - 1.0))
    E = e + 0.5 * u**2
    
    U = np.zeros((3, N))
    U[0, :] = rho
    U[1, :] = rho * u
    U[2, :] = rho * E
    
    # Compute sources
    S = model.source(uu=U, gamma=gamma_arr)
    
    print("Momentum source without pressure gradient:")
    print(S[1])
    
    # Since u=0, the source should just be the gravity term (and tau_w=0)
    # S[1] should be: - rho * g * sin(theta) * A? 
    # Wait, __g1 was implemented as:
    # (p_x / area_x * dela_delx - tau_w_x * l + mdot_w_x * ... - uu0 * g * sin(theta))
    # Note: the equation for __g1 we implemented has `- uu0 * g_gravity * np.sin(elev_angle)`.
    # Let's see what S[1] evaluates to.
    
    expected_grav_source = - rho * 9.81 * np.sin(np.pi / 2.0)
    print("Expected grav source:", expected_grav_source)
    
    assert np.allclose(S[1], expected_grav_source)
    print("Gravity verification passed!")

if __name__ == '__main__':
    main()
