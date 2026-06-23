import numpy as np
import sys
import os

# Add parent directory to path to import cfdlite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cfdlite.onedim import PipeModel, FlowSolver

def main():
    # Setup simple geometry
    x = np.linspace(0, 1, 10)
    area = np.ones_like(x) * 0.1
    perimeter = np.ones_like(x) * 0.4
    
    gamma = 1.4
    R_gas = 287.05
    
    # Static properties for a simple flow
    p = 100000.0  # 100 kPa
    T = 300.0
    u = 50.0
    
    rho = p / (R_gas * T)
    e = p / (rho * (gamma - 1.0))
    E = e + 0.5 * u**2
    
    # Create U vector
    U = np.zeros((3, len(x)))
    U[0, :] = rho
    U[1, :] = rho * u
    U[2, :] = rho * E
    
    # Orifice injection properties
    # Injecting choked flow in the middle cells
    P0_inj = np.zeros_like(x)
    T0_inj = np.zeros_like(x)
    A_inj = np.zeros_like(x)
    theta_inj = np.full_like(x, np.pi/4)  # 45 degrees injection
    
    # Inject at indices 4 and 5
    P0_inj[4:6] = 200000.0  # 200 kPa (choked since p/P0 = 0.5 < 0.528)
    T0_inj[4:6] = 300.0
    A_inj[4:6] = 0.01  # some small area
    
    model = PipeModel(
        x=x,
        area=area,
        perimeter=perimeter,
        P0_inj=P0_inj,
        T0_inj=T0_inj,
        A_inj=A_inj,
        theta_inj=theta_inj
    )
    
    gamma_arr = np.ones_like(x) * gamma
    
    # Get source terms
    S = model.source(uu=U, gamma=gamma_arr)
    
    print("Mass source:")
    print(S[0])
    print("Momentum source:")
    print(S[1])
    print("Energy source:")
    print(S[2])
    
    # Check that injection only occurs at indices 4 and 5
    assert np.all(S[0, :4] == 0)
    assert np.all(S[0, 6:] == 0)
    assert np.all(S[0, 4:6] > 0)
    
    # Check momentum source is positive due to theta_inj = pi/4
    assert np.all(S[1, 4:6] > 0)
    
    print("Verification passed!")

if __name__ == "__main__":
    main()
