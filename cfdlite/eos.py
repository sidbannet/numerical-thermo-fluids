import numpy as np
from abc import ABC, abstractmethod


class AbstractEOS(ABC):
    """Abstract base class for an Equation of State (EOS)."""

    @abstractmethod
    def pressure(self, rho: np.ndarray, e: np.ndarray, **kwargs) -> np.ndarray:
        """Compute static pressure from density and internal energy."""
        pass

    @abstractmethod
    def temperature(
            self,
            rho: np.ndarray,
            e: np.ndarray,
            **kwargs) -> np.ndarray:
        """Compute static temperature from density and internal energy."""
        pass

    @abstractmethod
    def speed_of_sound(
            self,
            rho: np.ndarray,
            e: np.ndarray,
            **kwargs) -> np.ndarray:
        """Compute the local speed of sound from density and internal energy."""
        pass


class IdealGasEOS(AbstractEOS):
    """Ideal Gas Equation of State."""

    def __init__(self, R_gas: float = 287.05):
        self.R_gas = R_gas

    def pressure(
            self,
            rho: np.ndarray,
            e: np.ndarray,
            gamma: np.ndarray = None,
            **kwargs) -> np.ndarray:
        """p = (gamma - 1) * rho * e"""
        if gamma is None:
            raise ValueError("gamma is required for IdealGasEOS")
        return (gamma - 1.0) * rho * e

    def temperature(
            self,
            rho: np.ndarray,
            e: np.ndarray,
            gamma: np.ndarray = None,
            **kwargs) -> np.ndarray:
        """T = p / (rho * R_gas)"""
        p = self.pressure(rho, e, gamma=gamma)
        # Avoid division by zero
        return np.divide(
            p,
            rho * self.R_gas,
            out=np.zeros_like(p),
            where=rho > 0)

    def speed_of_sound(
            self,
            rho: np.ndarray,
            e: np.ndarray,
            gamma: np.ndarray = None,
            **kwargs) -> np.ndarray:
        """a = sqrt(gamma * R * T) = sqrt(gamma * p / rho)"""
        p = self.pressure(rho, e, gamma=gamma)
        return np.sqrt(
            np.maximum(
                0.0,
                gamma *
                np.divide(
                    p,
                    rho,
                    out=np.zeros_like(p),
                    where=rho > 0)))


class CoolPropEOS(AbstractEOS):
    """Equation of state using CoolProp for real fluids."""

    def __init__(self, fluid: str = 'Water', backend: str = 'HEOS'):
        self.fluid = fluid if backend == 'HEOS' else f"{backend}::{fluid}"
        try:
            import CoolProp.CoolProp as CP  # type: ignore
            self.CP = CP
        except ImportError:
            raise ImportError(
                "CoolProp is not installed. Please install it to use CoolPropEOS.")

    def pressure(self, rho: np.ndarray, e: np.ndarray, **kwargs) -> np.ndarray:
        """Compute pressure using CoolProp."""
        is_scalar = np.ndim(rho) == 0
        rho_flat = np.atleast_1d(rho).astype(float)
        e_flat = np.atleast_1d(e).astype(float)
        p = self.CP.PropsSI('P', 'D', rho_flat, 'U', e_flat, self.fluid)
        p_arr = np.atleast_1d(p)
        return float(p_arr[0]) if is_scalar else p_arr.reshape(np.shape(rho))

    def temperature(
            self,
            rho: np.ndarray,
            e: np.ndarray,
            **kwargs) -> np.ndarray:
        """Compute temperature using CoolProp."""
        is_scalar = np.ndim(rho) == 0
        rho_flat = np.atleast_1d(rho).astype(float)
        e_flat = np.atleast_1d(e).astype(float)
        t = self.CP.PropsSI('T', 'D', rho_flat, 'U', e_flat, self.fluid)
        t_arr = np.atleast_1d(t)
        return float(t_arr[0]) if is_scalar else t_arr.reshape(np.shape(rho))

    def _hem_speed_of_sound(
            self,
            P_array: np.ndarray,
            s_array: np.ndarray) -> np.ndarray:
        """Calculate the Homogeneous Equilibrium Speed of Sound using an isentropic compressibility approach."""
        # This will operate on arrays of properties that are confirmed to be in
        # the two-phase region.
        delta_p = 100.0  # Pa perturbation

        # State 1 (P + dP)
        P1 = P_array + delta_p
        s_f1 = self.CP.PropsSI('S', 'P', P1, 'Q', 0, self.fluid)
        s_g1 = self.CP.PropsSI('S', 'P', P1, 'Q', 1, self.fluid)
        x1 = np.clip((s_array - s_f1) / (s_g1 - s_f1 + 1e-12), 0.0, 1.0)
        v_f1 = 1.0 / self.CP.PropsSI('D', 'P', P1, 'Q', 0, self.fluid)
        v_g1 = 1.0 / self.CP.PropsSI('D', 'P', P1, 'Q', 1, self.fluid)
        v1 = x1 * v_g1 + (1.0 - x1) * v_f1

        # State 2 (P - dP)
        P2 = np.maximum(P_array - delta_p, 100.0)  # avoid negative pressure
        s_f2 = self.CP.PropsSI('S', 'P', P2, 'Q', 0, self.fluid)
        s_g2 = self.CP.PropsSI('S', 'P', P2, 'Q', 1, self.fluid)
        x2 = np.clip((s_array - s_f2) / (s_g2 - s_f2 + 1e-12), 0.0, 1.0)
        v_f2 = 1.0 / self.CP.PropsSI('D', 'P', P2, 'Q', 0, self.fluid)
        v_g2 = 1.0 / self.CP.PropsSI('D', 'P', P2, 'Q', 1, self.fluid)
        v2 = x2 * v_g2 + (1.0 - x2) * v_f2

        dv_dp = (v1 - v2) / (P1 - P2)
        v = 0.5 * (v1 + v2)

        # a = v / sqrt(-(dv/dP)_s)
        a2 = - (v**2) / np.minimum(dv_dp, -1e-12)
        return np.sqrt(np.maximum(a2, 1e-6))

    def speed_of_sound(
            self,
            rho: np.ndarray,
            e: np.ndarray,
            **kwargs) -> np.ndarray:
        """Compute speed of sound handling both single-phase and two-phase regimes."""
        is_scalar = np.ndim(rho) == 0
        rho_flat = np.atleast_1d(rho).astype(float)
        e_flat = np.atleast_1d(e).astype(float)

        # First query the quality to identify phase regions
        # CoolProp returns -1 for single-phase states
        Q = self.CP.PropsSI('Q', 'D', rho_flat, 'U', e_flat, self.fluid)
        Q_arr = np.atleast_1d(Q)

        # Identify indices
        idx_2phase = np.where((Q_arr >= 0.0) & (Q_arr <= 1.0))[0]
        idx_1phase = np.where((Q_arr < 0.0) | (Q_arr > 1.0))[0]

        a = np.zeros_like(rho_flat)

        # Single Phase calculation
        if len(idx_1phase) > 0:
            a[idx_1phase] = self.CP.PropsSI(
                'A', 'D', rho_flat[idx_1phase], 'U', e_flat[idx_1phase], self.fluid)

        # Two Phase calculation (HEM fallback)
        if len(idx_2phase) > 0:
            P_2ph = self.CP.PropsSI(
                'P',
                'D',
                rho_flat[idx_2phase],
                'U',
                e_flat[idx_2phase],
                self.fluid)
            S_2ph = self.CP.PropsSI(
                'S',
                'D',
                rho_flat[idx_2phase],
                'U',
                e_flat[idx_2phase],
                self.fluid)
            a[idx_2phase] = self._hem_speed_of_sound(P_2ph, S_2ph)

        return a[0] if is_scalar else a.reshape(rho.shape)
