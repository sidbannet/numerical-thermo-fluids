import numpy as np
import CoolProp.CoolProp as CP
from cfdlite.eos import CoolPropEOS

def test_single_phase():
    print("--- Single Phase Test ---")
    eos = CoolPropEOS(fluid='Water')
    # Standard conditions: Liquid water at 1 atm, 300K
    rho = 996.5  # kg/m^3
    e = 112500.0 # J/kg (approx)
    
    p = eos.pressure(rho, e)
    t = eos.temperature(rho, e)
    a = eos.speed_of_sound(rho, e)
    
    print(f"Density: {rho} kg/m^3, Internal Energy: {e} J/kg")
    print(f"Computed Pressure: {p:.2f} Pa")
    print(f"Computed Temperature: {t:.2f} K")
    print(f"Computed Speed of Sound: {a:.2f} m/s")
    
def test_two_phase():
    print("\n--- Two-Phase Test (HEM Speed of Sound) ---")
    eos = CoolPropEOS(fluid='Water')
    
    # Let's create a two-phase mixture at 1 atm (101325 Pa)
    p_sat = 101325.0
    v_f = 1.0 / CP.PropsSI('D', 'P', p_sat, 'Q', 0, 'Water')
    v_g = 1.0 / CP.PropsSI('D', 'P', p_sat, 'Q', 1, 'Water')
    e_f = CP.PropsSI('U', 'P', p_sat, 'Q', 0, 'Water')
    e_g = CP.PropsSI('U', 'P', p_sat, 'Q', 1, 'Water')
    
    x = 0.5 # 50% quality by mass
    v_m = x * v_g + (1 - x) * v_f
    e_m = x * e_g + (1 - x) * e_f
    rho_m = 1.0 / v_m
    
    p = eos.pressure(rho_m, e_m)
    t = eos.temperature(rho_m, e_m)
    a = eos.speed_of_sound(rho_m, e_m)
    
    print(f"Quality: {x}, Density: {rho_m:.2f} kg/m^3, Internal Energy: {e_m:.2f} J/kg")
    print(f"Computed Pressure: {p:.2f} Pa (Expected: {p_sat})")
    print(f"Computed Temperature: {t:.2f} K")
    print(f"Computed HEM Speed of Sound: {a:.2f} m/s")

def test_arrays():
    print("\n--- Array Vectorization Test ---")
    eos = CoolPropEOS(fluid='Water')
    # Mix of liquid, two-phase, and vapor
    rho = np.array([996.5, 5.0, 0.5])
    e = np.array([112500.0, 1.5e6, 2.8e6])
    
    p = eos.pressure(rho, e)
    a = eos.speed_of_sound(rho, e)
    
    print("Densities:", rho)
    print("Pressures:", p)
    print("Speed of sound:", a)

if __name__ == '__main__':
    test_single_phase()
    test_two_phase()
    test_arrays()
