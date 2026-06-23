import numpy as np
from abc import ABC, abstractmethod

class AbstractEOS(ABC):
    """Abstract base class for an Equation of State (EOS)."""

    @abstractmethod
    def pressure(self, rho: np.ndarray, e: np.ndarray, **kwargs) -> np.ndarray:
        """Compute static pressure from density and internal energy."""
        pass

    @abstractmethod
    def temperature(self, rho: np.ndarray, e: np.ndarray, **kwargs) -> np.ndarray:
        """Compute static temperature from density and internal energy."""
        pass

    @abstractmethod
    def speed_of_sound(self, rho: np.ndarray, e: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the local speed of sound from density and internal energy."""
        pass


class IdealGasEOS(AbstractEOS):
    """Ideal Gas Equation of State."""

    def __init__(self, R_gas: float = 287.05):
        self.R_gas = R_gas

    def pressure(self, rho: np.ndarray, e: np.ndarray, gamma: np.ndarray = None, **kwargs) -> np.ndarray:
        """p = (gamma - 1) * rho * e"""
        if gamma is None:
            raise ValueError("gamma is required for IdealGasEOS")
        return (gamma - 1.0) * rho * e

    def temperature(self, rho: np.ndarray, e: np.ndarray, gamma: np.ndarray = None, **kwargs) -> np.ndarray:
        """T = p / (rho * R_gas)"""
        p = self.pressure(rho, e, gamma=gamma)
        # Avoid division by zero
        return np.divide(p, rho * self.R_gas, out=np.zeros_like(p), where=rho>0)

    def speed_of_sound(self, rho: np.ndarray, e: np.ndarray, gamma: np.ndarray = None, **kwargs) -> np.ndarray:
        """a = sqrt(gamma * R * T) = sqrt(gamma * p / rho)"""
        p = self.pressure(rho, e, gamma=gamma)
        return np.sqrt(np.maximum(0.0, gamma * np.divide(p, rho, out=np.zeros_like(p), where=rho>0)))
