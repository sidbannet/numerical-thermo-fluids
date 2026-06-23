import unittest
import numpy as np
# pyrefly: ignore [missing-import]
import CoolProp.CoolProp as CP
from cfdlite.eos import CoolPropEOS


class TestCoolPropEOS(unittest.TestCase):
    def setUp(self):
        self.eos = CoolPropEOS(fluid='Water')

    def test_single_phase(self):
        # Liquid water at ~1 atm, 300K
        rho = 996.5  # kg/m^3
        e = 112500.0  # J/kg

        p = self.eos.pressure(rho, e)
        t = self.eos.temperature(rho, e)
        a = self.eos.speed_of_sound(rho, e)

        # We just assert it calculates without error and returns reasonable
        # values
        self.assertGreater(p, 0.0)
        self.assertGreater(t, 273.15)
        self.assertGreater(a, 0.0)

    def test_two_phase_hem(self):
        # Two-phase mixture at 1 atm (101325 Pa)
        p_sat = 101325.0
        v_f = 1.0 / CP.PropsSI('D', 'P', p_sat, 'Q', 0, 'Water')
        v_g = 1.0 / CP.PropsSI('D', 'P', p_sat, 'Q', 1, 'Water')
        e_f = CP.PropsSI('U', 'P', p_sat, 'Q', 0, 'Water')
        e_g = CP.PropsSI('U', 'P', p_sat, 'Q', 1, 'Water')

        x = 0.5  # 50% quality by mass
        v_m = x * v_g + (1 - x) * v_f
        e_m = x * e_g + (1 - x) * e_f
        rho_m = 1.0 / v_m

        p = self.eos.pressure(rho_m, e_m)
        t = self.eos.temperature(rho_m, e_m)
        a = self.eos.speed_of_sound(rho_m, e_m)

        self.assertAlmostEqual(p, p_sat, delta=100.0)  # within 100 Pa
        self.assertGreater(a, 0.0)
        self.assertGreater(t, 273.15)

    def test_arrays(self):
        # Mix of liquid, two-phase, and vapor
        rho = np.array([996.5, 5.0, 0.5])
        e = np.array([112500.0, 1.5e6, 2.8e6])

        p = self.eos.pressure(rho, e)
        a = self.eos.speed_of_sound(rho, e)

        self.assertEqual(p.shape, rho.shape)
        self.assertEqual(a.shape, rho.shape)
        self.assertTrue(np.all(a > 0))


if __name__ == '__main__':
    unittest.main()
