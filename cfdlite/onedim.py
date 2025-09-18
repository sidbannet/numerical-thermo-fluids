"""
Solvers for one-dimensional computational fluid dynamics.

@author: siddhartha.banerjee
"""

import numpy as np
from enum import Enum, unique
from typing import Optional


class PipeModel:
    """Functions and properties of the CFD domain."""

    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        area: Optional[np.ndarray] = None,
        mdot_w: Optional[np.ndarray] = None,
        tau_w: Optional[np.ndarray] = None,
        q_w: Optional[np.ndarray] = None,
        perimeter: Optional[np.ndarray] = None,
    ) -> None:
        """Instantiate the class."""
        if x is None or area is None:
            self.x = np.array([])
            self.area = np.array([])
            self.dA_dx = np.array([])
        else:
            self.x = x
            self.area = area
            self.dA_dx = np.gradient(
                area, x,
                edge_order=1,
            )
        self.mdot_w = mdot_w if mdot_w is not None else np.array([])
        self.tau_w = tau_w if tau_w is not None else np.array([])
        self.q_w = q_w if q_w is not None else np.array([])
        self.perimeter = perimeter if perimeter is not None else np.array([])
        self.__ff0 = np.vectorize(
            lambda uu0, uu1, uu2, gamma: uu1
        )
        self.__ff1 = np.vectorize(
            lambda uu0, uu1, uu2, gamma: (uu1 ** 2) / uu0 + (
                uu0 * (uu2 / uu0 - 0.5 * (uu1 / uu0) ** 2) * (gamma - 1)
            )
        )
        self.__ff2 = np.vectorize(
            lambda uu0, uu1, uu2, gamma: uu1 * (
                gamma * uu2 / uu0 - (gamma - 1) / 2 * (uu1 / uu0) ** 2
            )
        )
        self.__g0 = np.vectorize(
            lambda uu0, uu1, uu2, mdot_w_x, length: mdot_w_x * length
        )
        self.__g1 = np.vectorize(
            lambda uu0, uu1, uu2, gamma, area_x, dela_delx, tau_w_x, l: (
                uu2 / uu0 - 0.5 * (uu1 / uu0) ** 2
            ) * uu0 / area_x * (gamma - 1) * dela_delx - tau_w_x * l
        )
        self.__g2 = np.vectorize(
            lambda uu0, uu1, uu2, q_x, l: q_x * l
        )
        self.__velocity = np.vectorize(
            lambda uu0, uu1, uu2: uu1 / uu0
        )
        self.__density = np.vectorize(
            lambda uu0, uu1, uu2, area_x: uu0 / area_x
        )
        self.__internal_energy = np.vectorize(
            lambda uu0, uu1, uu2: (uu2 / uu0) - (
                0.5 * (uu1 / uu0) ** 2
            )
        )
        self.__pressure = np.vectorize(
            lambda uu0, uu1, uu2, area_x, gamma: (
                self.__internal_energy(
                    uu0, uu1, uu2)
            ) * (
                self.__density(uu0, uu1, uu2, area_x)
            ) * (gamma - 1)
        )

    def flux(
        self,
        uu: np.ndarray = np.nan,
        gamma: np.ndarray = np.nan,
    ) -> np.array:
        """Get flux vector."""
        return np.array(
            [
                self.__ff0(uu[0], uu[1], uu[2], gamma),
                self.__ff1(uu[0], uu[1], uu[2], gamma),
                self.__ff2(uu[0], uu[1], uu[2], gamma),
            ],
        )

    def source(
        self,
        uu: np.array = np.nan,
        gamma: np.array = np.nan,
    ) -> np.array:
        """Get source vector."""
        return np.array(
            [
                self.__g0(uu[0], uu[1], uu[2], self.mdot_w, self.perimeter),
                self.__g1(uu[0], uu[1], uu[2], gamma, self.area, self.dA_dx,
                          self.tau_w, self.perimeter),
                self.__g2(uu[0], uu[1], uu[2], self.q_w, self.perimeter),
            ],
        )

    def thermo(
        self,
        uu: np.array = np.nan,
        gamma: np.array = np.nan,
    ) -> dict:
        """Get the thermodynamic properties."""
        return dict(
            {
                'velocity': self.__velocity(uu[0], uu[1], uu[2]),
                'density': self.__density(uu[0], uu[1], uu[2], self.area),
                'pressure': self.__pressure(uu[0], uu[1], uu[2], self.area,
                                            gamma),
                'internal_energy': self.__internal_energy(uu[0], uu[1], uu[2]),
            }
        )

class FlowSolver:
    """Finite Volume scheme for non-linear inhomogeneous transport."""
    """Finite Volume scheme for non-linear inhomogeneous transport."""

    def __init__(
        self,
        model: Optional[PipeModel] = None,
        initial_solution: Optional[np.ndarray] = None,
        intake_boundary: Optional[dict] = None,
        outlet_boundary: Optional[dict] = None,
    ):
        """Instantiate the class."""
        self.model = model
        self.uu = initial_solution if initial_solution is not None else np.nan
        self.gamma = (
            np.full_like(initial_solution[0], np.nan)
            if initial_solution is not None else np.nan
        )
        if intake_boundary is None:
            self.intake_boundary = {
                BoundaryCategory.MF: {BoundaryType.DI: float(0)},
                BoundaryCategory.TP: {BoundaryType.NE: float(0)},
                BoundaryCategory.TH: {BoundaryType.DI: float(0)},
            }
        else:
            self.intake_boundary = intake_boundary
        if outlet_boundary is None:
            self.outlet_boundary = {
                BoundaryCategory.MF: {BoundaryType.NE: float(0)},
                BoundaryCategory.TP: {BoundaryType.NE: float(0)},
                BoundaryCategory.TH: {BoundaryType.NE: float(0)},
            }
        else:
            self.outlet_boundary = outlet_boundary

    def __get_boundary_flux_for_predictor(
        self,
        uu: np.array = np.nan,
        gamma: np.array = np.nan,
    ) -> np.array:
        """Get boundary flux terms for predictor."""
        x = self.model.x
        flux = self.model.flux(uu=uu, gamma=gamma)
        outlet = self.outlet_boundary
        mfr_outlet = np.nan
        total_pressure_outlet = np.nan
        total_enthalpy_outlet = np.nan
        scale = [
            1,
            self.model.area[-1],
            uu[1][-1],
        ]
        offset = [
            0,
            0.5 * (uu[1][-1] ** 2.0) / uu[0][-1],
            0,
        ]
        if outlet[BoundaryCategory.MF].keys().__contains__(BoundaryType.DI):
            mfr_outlet = np.fromiter(
                outlet[BoundaryCategory.MF].values(),
                dtype=float,
            )[0] * scale[0] + offset[0]
        elif outlet[BoundaryCategory.MF].keys().__contains__(BoundaryType.NE):
            mfr_outlet = flux[0][-1] + np.fromiter(
                outlet[BoundaryCategory.MF].values(),
                dtype=float,
            )[0] * (x[-1] - x[-2]) * scale[0] + offset[0]
        if outlet[BoundaryCategory.TP].keys().__contains__(BoundaryType.DI):
            total_pressure_outlet = np.fromiter(
                outlet[BoundaryCategory.TP].values(),
                dtype=float,
            )[0] * scale[1] + offset[1]
        elif outlet[BoundaryCategory.TP].keys().__contains__(BoundaryType.NE):
            total_pressure_outlet = flux[1][-1] + np.fromiter(
                outlet[BoundaryCategory.TP].values(),
                dtype=float,
            )[0] * (x[-1] - x[-2]) * scale[1] + offset[1]
        if outlet[BoundaryCategory.TH].keys().__contains__(BoundaryType.DI):
            total_enthalpy_outlet = np.fromiter(
                outlet[BoundaryCategory.TH].values(),
                dtype=float,
            )[0] * scale[2] + offset[2]
        elif outlet[BoundaryCategory.TH].keys().__contains__(BoundaryType.NE):
            total_enthalpy_outlet = flux[2][-1] + np.fromiter(
                outlet[BoundaryCategory.TH].values(),
                dtype=float,
            )[0] * (x[-1] - x[-2]) * scale[2] + offset[2]
        return np.array(
            [
                mfr_outlet,
                total_pressure_outlet,
                total_enthalpy_outlet,
            ]
        )

    def __get_boundary_flux_for_corrector(
        self,
        uu: np.array = np.nan,
        gamma: np.array = np.nan,
    ) -> np.array:
        """Get boundary flux terms for corrector."""
        x = self.model.x
        flux = self.model.flux(uu=uu, gamma=gamma)
        intake = self.intake_boundary
        mfr_intake = np.nan
        total_pressure_intake = np.nan
        total_enthalpy_intake = np.nan
        scale = [
            1,
            self.model.area[0],
            uu[1][0],
        ]
        offset = [
            0,
            0.5 * (uu[1][0] ** 2.0) / uu[0][0],
            0,
        ]
        if intake[BoundaryCategory.MF].keys().__contains__(BoundaryType.DI):
            mfr_intake = np.fromiter(
                intake[BoundaryCategory.MF].values(),
                dtype=float,
            )[0] * scale[0] + offset[0]
        elif intake[BoundaryCategory.MF].keys().__contains__(BoundaryType.NE):
            mfr_intake = flux[0][0] - np.fromiter(
                intake[BoundaryCategory.MF].values(),
                dtype=float,
            )[0] * (x[1] - x[0]) * scale[0] - offset[0]
        if intake[BoundaryCategory.TP].keys().__contains__(BoundaryType.DI):
            total_pressure_intake = np.fromiter(
                intake[BoundaryCategory.TP].values(),
                dtype=float,
            )[0] * scale[1] + offset[1]
        elif intake[BoundaryCategory.TP].keys().__contains__(BoundaryType.NE):
            total_pressure_intake = flux[1][0] - np.fromiter(
                intake[BoundaryCategory.TP].values(),
                dtype=float,
            )[0] * (x[1] - x[0]) * scale[1] - offset[1]
        if intake[BoundaryCategory.TH].keys().__contains__(BoundaryType.DI):
            total_enthalpy_intake = np.fromiter(
                intake[BoundaryCategory.TH].values(),
                dtype=float,
            )[0] * scale[2] + offset[2]
        elif intake[BoundaryCategory.TH].keys().__contains__(BoundaryType.NE):
            total_enthalpy_intake = flux[2][0] - np.fromiter(
                intake[BoundaryCategory.TH].values(),
                dtype=float,
            )[0] * (x[1] - x[0]) * scale[2] - offset[2]
        return np.array(
            [
                mfr_intake,
                total_pressure_intake,
                total_enthalpy_intake,
            ]
        )

    @staticmethod
    def __predictor(
        x: np.ndarray = np.nan,
        flux: np.ndarray = np.nan,
        source: np.ndarray = np.nan,
        outlet_flux: np.ndarray = np.nan,
    ) -> np.array:
        """Predictor step.
        :type outlet_flux: numpy.ndarray
        """
        return - np.diff(
            np.append(flux.T, np.array([outlet_flux]), axis=0).T,
            n=int(1),
            axis=1,
        ) / np.diff(
            x,
            n=int(1),
            append=2 * x[-1] - x[-2],
        ) + source

    @staticmethod
    def __corrector(
        x: np.ndarray = np.nan,
        flux: np.ndarray = np.nan,
        source: np.ndarray = np.nan,
        intake_flux: np.ndarray = np.nan,
    ) -> np.array:
        """Corrector step.
        :type intake_flux: numpy.ndarray
        """
        return - np.diff(
            np.append(np.array([intake_flux]), flux.T, axis=0).T,
            n=int(1),
            axis=1,
        ) / np.diff(
            np.insert(arr=x, obj=0, values=2 * x[0] - x[1]),
            n=int(1),
        ) + source

    def update(
        self,
        dt: float = 1e-6,
    ) -> None:
        """Update the solution for next time step."""
        # TODO: Update Gamma and Pass on Inlet and Outlet boundaries
        uu_predictor = self.uu + dt * (
            self.__predictor(
                x=self.model.x,
                flux=self.model.flux(uu=self.uu, gamma=np.asarray(self.gamma)),
                source=self.model.source(uu=self.uu, gamma=self.gamma),
                outlet_flux=self.__get_boundary_flux_for_predictor(
                    uu=self.uu, gamma=self.gamma),
            )
        )
        uu_corrector = self.uu + dt * (
            self.__corrector(
                x=self.model.x,
                flux=self.model.flux(uu=uu_predictor, gamma=self.gamma),
                source=self.model.source(uu=uu_predictor, gamma=self.gamma),
                intake_flux=self.__get_boundary_flux_for_corrector(
                    uu=uu_predictor, gamma=self.gamma),
            )
        )
        self.uu = 0.5 * (uu_predictor + uu_corrector)


@unique
class BoundaryType(Enum):
    DI = 'Specified Value (Dirichlet)'
    NE = 'Specified Normal Gradient (Neumann)'
    DR = 'Derived Boundary'


@unique
class BoundaryCategory(Enum):
    MF = 'Mass flow rate'
    TP = 'Total Pressure'
    TH = 'Total Enthalpy'
