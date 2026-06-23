import unittest
import numpy as np
from cfdlite.onedim import PipeModel


class TestOrificeInjection(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1, 10)
        self.area = np.ones_like(self.x) * 0.1
        self.perimeter = np.ones_like(self.x) * 0.4
        
        self.gamma = 1.4
        self.R_gas = 287.05
        
        # Static properties for a simple flow
        p = 100000.0  # 100 kPa
        T = 300.0
        u = 50.0
        
        rho = p / (self.R_gas * T)
        e = p / (rho * (self.gamma - 1.0))
        E = e + 0.5 * u**2
        
        # Create U vector
        self.U = np.zeros((3, len(self.x)))
        self.U[0, :] = rho
        self.U[1, :] = rho * u
        self.U[2, :] = rho * E
        self.gamma_arr = np.ones_like(self.x) * self.gamma

    def test_orifice_injection_source_terms(self):
        # Orifice injection properties
        # Injecting choked flow in the middle cells
        P0_inj = np.zeros_like(self.x)
        T0_inj = np.zeros_like(self.x)
        A_inj = np.zeros_like(self.x)
        theta_inj = np.full_like(self.x, np.pi/4)  # 45 degrees injection
        
        # Inject at indices 4 and 5
        P0_inj[4:6] = 200000.0  # 200 kPa (choked since p/P0 = 0.5 < 0.528)
        T0_inj[4:6] = 300.0
        A_inj[4:6] = 0.01  # some small area
        
        model = PipeModel(
            x=self.x,
            area=self.area,
            perimeter=self.perimeter,
            P0_inj=P0_inj,
            T0_inj=T0_inj,
            A_inj=A_inj,
            theta_inj=theta_inj
        )
        
        # Get source terms
        S = model.source(uu=self.U, gamma=self.gamma_arr)
        
        # Check that injection only occurs at indices 4 and 5
        self.assertTrue(np.all(S[0, :4] == 0))
        self.assertTrue(np.all(S[0, 6:] == 0))
        self.assertTrue(np.all(S[0, 4:6] > 0))
        
        # Check momentum source is positive due to theta_inj = pi/4
        self.assertTrue(np.all(S[1, 4:6] > 0))
