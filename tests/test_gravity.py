import unittest
import numpy as np
from cfdlite.onedim import PipeModel


class TestGravitySource(unittest.TestCase):
    def test_vertical_pipe_hydrostatic_pressure(self):
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

        rho = np.ones_like(x) * 1000.0
        u = np.zeros_like(x)

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

        expected_grav_source = - rho * 9.81 * np.sin(np.pi / 2.0)

        np.testing.assert_allclose(S[1], expected_grav_source)


if __name__ == '__main__':
    unittest.main()
