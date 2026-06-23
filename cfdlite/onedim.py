"""
Solvers for one-dimensional computational fluid dynamics.

@author: siddhartha.banerjee
"""

import numpy as np
from enum import Enum, unique
from typing import Optional
from .eos import AbstractEOS, IdealGasEOS


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
        H_inj: Optional[np.ndarray] = None,
        P0_inj: Optional[np.ndarray] = None,
        T0_inj: Optional[np.ndarray] = None,
        A_inj: Optional[np.ndarray] = None,
        theta_inj: Optional[np.ndarray] = None,
        gamma_inj: Optional[np.ndarray] = None,
        R_gas: float = 287.05,
        g_gravity: float = 9.81,
        elev_angle: Optional[np.ndarray] = None,
        eos: Optional[AbstractEOS] = None,
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
        self.mdot_w = mdot_w if mdot_w is not None else np.zeros_like(self.x)
        self.tau_w = tau_w if tau_w is not None else np.zeros_like(self.x)
        self.q_w = q_w if q_w is not None else np.zeros_like(self.x)
        self.perimeter = perimeter if perimeter is not None else np.zeros_like(
            self.x)
        self.H_inj = H_inj if H_inj is not None else np.zeros_like(self.x)
        self.P0_inj = P0_inj if P0_inj is not None else np.array([])
        self.T0_inj = T0_inj if T0_inj is not None else np.array([])
        self.A_inj = A_inj if A_inj is not None else np.array([])
        self.theta_inj = theta_inj if theta_inj is not None else (
            np.full_like(
                self.x,
                np.pi /
                2.0) if len(
                self.x) > 0 else np.array(
                []))
        self.gamma_inj = gamma_inj if gamma_inj is not None else np.array([])
        self.R_gas = R_gas
        self.g_gravity = g_gravity
        self.elev_angle = elev_angle if elev_angle is not None else np.zeros_like(
            self.x)
        self.eos = eos if eos is not None else IdealGasEOS(R_gas=self.R_gas)
        self.__ff0 = np.vectorize(
            lambda uu0, uu1, uu2, gamma: uu1
        )
        self.__ff1 = lambda uu0, uu1, uu2, gamma: (
            uu1 ** 2) / uu0 + self.__pressure(uu0, uu1, uu2, gamma)
        self.__ff2 = lambda uu0, uu1, uu2, gamma: uu1 / \
            uu0 * (uu2 + self.__pressure(uu0, uu1, uu2, gamma))
        self.__g0 = np.vectorize(
            lambda uu0, uu1, uu2, mdot_w_x, length: mdot_w_x * length
        )
        self.__g1 = np.vectorize(
            lambda uu0,
            uu1,
            uu2,
            p_x,
            area_x,
            dela_delx,
            tau_w_x,
            l,
            mdot_w_x,
            u_inj_x,
            theta_inj_x,
            elev_angle_x: (
                p_x /
                area_x *
                dela_delx -
                tau_w_x *
                l +
                mdot_w_x *
                u_inj_x *
                np.cos(theta_inj_x) *
                l -
                uu0 *
                self.g_gravity *
                np.sin(elev_angle_x)))
        self.__g2 = np.vectorize(
            lambda uu0,
            uu1,
            uu2,
            q_x,
            mdot_w_x,
            H_inj_x,
            l,
            elev_angle_x: (
                q_x *
                l +
                mdot_w_x *
                H_inj_x *
                l -
                uu1 *
                self.g_gravity *
                np.sin(elev_angle_x)))
        self.__velocity = np.vectorize(
            lambda uu0, uu1, uu2: uu1 / uu0
        )
        self.__density = np.vectorize(
            lambda uu0, uu1, uu2: uu0
        )
        self.__internal_energy = np.vectorize(
            lambda uu0, uu1, uu2: (uu2 / uu0) - (
                0.5 * (uu1 / uu0) ** 2
            )
        )
        self.__pressure = lambda uu0, uu1, uu2, gamma: self.eos.pressure(
            rho=self.__density(uu0, uu1, uu2),
            e=self.__internal_energy(uu0, uu1, uu2),
            gamma=gamma
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

    def _compute_dynamic_injection(
            self,
            p: np.ndarray,
            gamma_main: np.ndarray) -> tuple:
        """Dynamically compute mdot_w, u_inj, and H_inj using compressible orifice flow."""
        g = self.gamma_inj if len(self.gamma_inj) > 0 else gamma_main
        R = self.R_gas
        P0 = self.P0_inj
        T0 = self.T0_inj
        A = self.A_inj

        p_ratio = np.divide(p, P0, out=np.ones_like(p), where=P0 > 0)
        critical_ratio = (2.0 / (g + 1.0)) ** (g / (g - 1.0))

        is_choked = p_ratio <= critical_ratio

        mach_sq = np.where(is_choked, 1.0, np.maximum(
            0.0, (2.0 / (g - 1.0)) * (p_ratio ** (-(g - 1.0) / g) - 1.0)))
        mach_inj = np.sqrt(mach_sq)

        T_inj = T0 / (1.0 + 0.5 * (g - 1.0) * mach_sq)
        rho_inj = np.divide(P0,
                            R * T0,
                            out=np.zeros_like(P0),
                            where=T0 > 0) * (np.divide(T_inj,
                                                       T0,
                                                       out=np.zeros_like(T_inj),
                                                       where=T0 > 0)) ** (1.0 / (g - 1.0))

        a_inj = np.sqrt(g * R * T_inj)
        u_inj = mach_inj * a_inj

        mdot_w = np.where(P0 > p, rho_inj * u_inj * A, 0.0)
        u_inj = np.where(P0 > p, u_inj, 0.0)
        H_inj = (g * R / (g - 1.0)) * T0

        return mdot_w, u_inj, H_inj

    def source(
        self,
        uu: np.array = np.nan,
        gamma: np.array = np.nan,
    ) -> np.array:
        """Get source vector."""
        p = self.__pressure(uu[0], uu[1], uu[2], gamma)
        if len(self.P0_inj) > 0:
            mdot_w, u_inj, H_inj = self._compute_dynamic_injection(p, gamma)
        else:
            mdot_w = self.mdot_w
            u_inj = np.zeros_like(self.x) if len(self.x) > 0 else np.array([])
            H_inj = self.H_inj

        return np.array([self.__g0(uu[0],
                                   uu[1],
                                   uu[2],
                                   mdot_w,
                                   self.perimeter),
                         self.__g1(uu[0],
                                   uu[1],
                                   uu[2],
                                   p,
                                   self.area,
                                   self.dA_dx,
                                   self.tau_w,
                                   self.perimeter,
                                   mdot_w,
                                   u_inj,
                                   self.theta_inj,
                                   self.elev_angle),
                         self.__g2(uu[0],
                                   uu[1],
                                   uu[2],
                                   self.q_w,
                                   mdot_w,
                                   H_inj,
                                   self.perimeter,
                                   self.elev_angle),
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
                'density': self.__density(uu[0], uu[1], uu[2]),
                'pressure': self.__pressure(uu[0], uu[1], uu[2], gamma),
                'internal_energy': self.__internal_energy(uu[0], uu[1], uu[2]),
            }
        )


class FlowSolver:
    """Finite Volume scheme for non-linear inhomogeneous transport (MacCormack)."""

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


@unique
class LimiterType(Enum):
    """Flux-limiter choices for TVD / MUSCL reconstruction."""
    MINMOD = 'minmod'
    VAN_LEER = 'van_leer'
    SUPERBEE = 'superbee'
    MC = 'monotonized_central'


class TVDSolver:
    """Second-order TVD finite-volume solver with MUSCL reconstruction.

    Uses slope-limited MUSCL reconstruction to form left/right interface
    states and Roe's approximate Riemann solver (with a local Lax-Friedrichs
    entropy fix) to compute face fluxes.  Time integration is performed with
    the SSP-RK2 (Heun) scheme so the overall scheme is formally second-order
    accurate in both space and time.

    The solver shares the same :class:`PipeModel` geometry and boundary
    convention as :class:`FlowSolver`, making it a drop-in upgrade when
    sharper shock resolution is needed.
    """

    def __init__(
        self,
        model: Optional[PipeModel] = None,
        initial_solution: Optional[np.ndarray] = None,
        intake_boundary: Optional[dict] = None,
        outlet_boundary: Optional[dict] = None,
        limiter: LimiterType = LimiterType.VAN_LEER,
    ) -> None:
        """Instantiate the TVD solver.

        Parameters
        ----------
        model:
            Pipe geometry / source-term model.
        initial_solution:
            Conservative-variable array of shape ``(3, N)``.
        intake_boundary:
            Left (inlet) boundary specification (same format as
            :class:`FlowSolver`).
        outlet_boundary:
            Right (outlet) boundary specification.
        limiter:
            Flux limiter to use for MUSCL slope reconstruction.
        """
        self.model = model
        self.uu = initial_solution if initial_solution is not None else np.nan
        self.gamma = (
            np.full_like(initial_solution[0], np.nan)
            if initial_solution is not None else np.nan
        )
        self.limiter = limiter
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

    # ------------------------------------------------------------------
    # Flux-limiter functions  phi(r)
    # ------------------------------------------------------------------

    @staticmethod
    def _limiter_minmod(r: np.ndarray) -> np.ndarray:
        """Minmod limiter: phi(r) = max(0, min(1, r))."""
        return np.maximum(0.0, np.minimum(1.0, r))

    @staticmethod
    def _limiter_van_leer(r: np.ndarray) -> np.ndarray:
        """Van Leer limiter: phi(r) = (r + |r|) / (1 + |r|)."""
        abs_r = np.abs(r)
        return (r + abs_r) / (1.0 + abs_r + 1e-30)

    @staticmethod
    def _limiter_superbee(r: np.ndarray) -> np.ndarray:
        """Superbee limiter: phi(r) = max(0, max(min(2r,1), min(r,2)))."""
        return np.maximum(
            0.0,
            np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)),
        )

    @staticmethod
    def _limiter_mc(r: np.ndarray) -> np.ndarray:
        """Monotonized-central (MC) limiter."""
        return np.maximum(
            0.0,
            np.minimum(
                np.minimum(2.0 * r, 0.5 * (1.0 + r)),
                2.0,
            ),
        )

    def _apply_limiter(self, r: np.ndarray) -> np.ndarray:
        """Dispatch to the selected limiter."""
        dispatch = {
            LimiterType.MINMOD: self._limiter_minmod,
            LimiterType.VAN_LEER: self._limiter_van_leer,
            LimiterType.SUPERBEE: self._limiter_superbee,
            LimiterType.MC: self._limiter_mc,
        }
        return dispatch[self.limiter](r)

    # ------------------------------------------------------------------
    # MUSCL reconstruction
    # ------------------------------------------------------------------

    def _muscl_reconstruct(
        self,
        uu: np.ndarray,
    ) -> tuple:
        """Return left and right interface states via MUSCL reconstruction.

        For a cell array ``uu`` of shape ``(3, N)`` the method returns
        ``(uL, uR)`` each of shape ``(3, N+1)`` representing states on the
        left and right side of each of the ``N+1`` cell faces (including
        the two ghost-cell faces at the domain boundaries).

        Ghost cells are populated using zero-gradient (Neumann) extrapolation
        so that boundary conditions can be imposed afterward.
        """
        n = uu.shape[1]

        # Extend with one ghost cell on each side (zero-gradient)
        u_ext = np.concatenate(
            [uu[:, :1], uu, uu[:, -1:]], axis=1
        )  # shape (3, N+2)

        delta = np.diff(u_ext, axis=1)            # shape (3, N+1): du_{i-1/2}
        delta_fwd = delta[:, 1:]                  # du_{i+1/2},  shape (3, N)
        delta_bwd = delta[:, :-1]                 # du_{i-1/2},  shape (3, N)

        # Smoothness ratio  r_i = du_{i-1/2} / du_{i+1/2}  (element-wise)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_fwd = np.where(
                np.abs(delta_fwd) > 1e-30,
                delta_bwd / delta_fwd,
                0.0,
            )
            r_bwd = np.where(
                np.abs(delta_bwd) > 1e-30,
                delta_fwd / delta_bwd,
                0.0,
            )

        phi_fwd = self._apply_limiter(r_fwd)   # limiter at cell i, right face
        phi_bwd = self._apply_limiter(r_bwd)   # limiter at cell i, left face

        # Reconstructed half-step slopes
        slope_L = 0.5 * phi_fwd * delta_fwd    # slope for right face of cell i
        slope_R = 0.5 * phi_bwd * delta_bwd    # slope for left  face of cell i

        # Face-state arrays; face k is between cell k-1 and cell k
        # uL[k] = state approaching face k from the left  (cell k-1 right side)
        # uR[k] = state approaching face k from the right (cell k   left  side)
        uL = np.empty((3, n + 1))
        uR = np.empty((3, n + 1))

        # Interior faces 1 .. N-1
        uL[:, 1:n] = uu[:, :-1] + slope_L[:, :-1]
        uR[:, 1:n] = uu[:, 1:] - slope_R[:, 1:]

        # Boundary faces: use first-order (zero-gradient ghost)
        uL[:, 0] = uu[:, 0]
        uR[:, 0] = uu[:, 0]
        uL[:, n] = uu[:, -1]
        uR[:, n] = uu[:, -1]

        return uL, uR

    # ------------------------------------------------------------------
    # Roe interface flux (with Harten entropy fix)
    # ------------------------------------------------------------------

    def _roe_flux(
        self,
        uL: np.ndarray,
        uR: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """Compute Roe's approximate Riemann flux at a set of interfaces.

        Parameters
        ----------
        uL, uR:
            Left / right conservative states, shape ``(3, M)``.
        gamma:
            Ratio of specific heats, shape ``(N,)``; a single representative
            value is taken at each interface.

        Returns
        -------
        F_roe:
            Interface fluxes, shape ``(3, M)``.
        """
        eps_entropy = 0.1   # Harten entropy-fix threshold (fraction of |a|)

        def _prim(u, gam):
            """Return (rho, v, p, H) from conservative state."""
            rho = u[0]
            v = u[1] / u[0]
            e = u[2] / u[0] - 0.5 * v ** 2
            if hasattr(self, 'model') and self.model.eos is not None:
                p = self.model.eos.pressure(rho, e, gamma=gam)
            else:
                p = rho * e * (gam - 1.0)
            H = (u[2] + p) / rho
            return rho, v, p, H

        def _flux_from_cons(u, gam):
            """Physical flux F(U)."""
            rho, v, p, H = _prim(u, gam)
            return np.array([
                u[1],
                u[1] * v + p,
                u[1] * H,
            ])

        # Interface-averaged gamma
        gam_face = 0.5 * (gamma[:-1] + gamma[1:])   # shape (M,)

        rhoL, vL, pL, HL = _prim(uL, gam_face)
        rhoR, vR, pR, HR = _prim(uR, gam_face)

        FL = _flux_from_cons(uL, gam_face)
        FR = _flux_from_cons(uR, gam_face)

        # Roe-averaged quantities
        sqrt_rhoL = np.sqrt(np.maximum(rhoL, 0.0))
        sqrt_rhoR = np.sqrt(np.maximum(rhoR, 0.0))
        denom = sqrt_rhoL + sqrt_rhoR + 1e-30

        v_roe = (sqrt_rhoL * vL + sqrt_rhoR * vR) / denom
        H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) / denom
        rho_roe = sqrt_rhoL * sqrt_rhoR

        if hasattr(
                self, 'model') and type(
                self.model.eos).__name__ != 'IdealGasEOS':
            # For general EOS, approximate Roe state internal energy
            eL = uL[2] / uL[0] - 0.5 * (uL[1] / uL[0]) ** 2
            eR = uR[2] / uR[0] - 0.5 * (uR[1] / uR[0]) ** 2
            e_roe = 0.5 * (eL + eR)
            a_roe = self.model.eos.speed_of_sound(
                rho_roe, e_roe, gamma=gam_face)
        else:
            a_roe = np.sqrt(np.maximum((gam_face - 1.0) *
                                       (H_roe - 0.5 * v_roe ** 2), 1e-30))

        # Eigenvalues
        lam1 = v_roe - a_roe
        lam2 = v_roe
        lam3 = v_roe + a_roe

        # Harten entropy fix
        def _fix(lam, lam_ref):
            abs_lam = np.abs(lam)
            delta = eps_entropy * np.abs(lam_ref)
            return np.where(abs_lam < delta, 0.5 *
                            (abs_lam ** 2 / delta + delta), abs_lam)

        abs_lam1 = _fix(lam1, a_roe)
        abs_lam2 = np.abs(lam2)
        abs_lam3 = _fix(lam3, a_roe)

        # Wave strengths via standard 1-D Euler Roe decomposition:
        #   alpha1 = (dp - rho_roe * a_roe * dv) / (2 * a_roe^2)
        #   alpha2 = drho - dp / a_roe^2
        #   alpha3 = (dp + rho_roe * a_roe * dv) / (2 * a_roe^2)
        dU = uR - uL  # noqa: F841  (kept for potential debug use)
        dp = pR - pL
        drho = rhoR - rhoL
        dv = vR - vL

        rho_roe = sqrt_rhoL * sqrt_rhoR
        a2 = a_roe ** 2 + 1e-30
        alpha1 = (dp - rho_roe * a_roe * dv) / (2.0 * a2)
        alpha2 = drho - dp / a2
        alpha3 = (dp + rho_roe * a_roe * dv) / (2.0 * a2)

        # Right eigenvectors (columns of R)
        r1 = np.array([np.ones_like(v_roe), v_roe -
                      a_roe, H_roe - v_roe * a_roe])
        r2 = np.array([np.ones_like(v_roe), v_roe, 0.5 * v_roe ** 2])
        r3 = np.array([np.ones_like(v_roe), v_roe +
                      a_roe, H_roe + v_roe * a_roe])

        # Roe dissipation
        dissipation = (
            abs_lam1 * alpha1 * r1
            + abs_lam2 * alpha2 * r2
            + abs_lam3 * alpha3 * r3
        )

        return 0.5 * (FL + FR) - 0.5 * dissipation

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    def _residual(
        self,
        uu: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """Compute dU/dt = -dF/dx + S for all interior cells.

        Parameters
        ----------
        uu:
            Conservative variable array, shape ``(3, N)``.
        gamma:
            Ratio of specific heats array, shape ``(N,)``.

        Returns
        -------
        rhs:
            Right-hand-side array, shape ``(3, N)``.
        """
        x = self.model.x
        dx = np.diff(x, append=2.0 * x[-1] - x[-2])

        uL, uR = self._muscl_reconstruct(uu)

        # Build a face-averaged gamma (length N+1); use nearest-cell values
        gamma_ext = np.concatenate([[gamma[0]], gamma, [gamma[-1]]])
        F_face = self._roe_flux(uL, uR, gamma_ext)   # shape (3, N+1)

        # Apply boundary conditions to leftmost / rightmost face fluxes
        #   - Left (intake) face: face index 0
        #   - Right (outlet) face: face index N
        self._apply_boundary_flux(F_face, uu, gamma)

        dF = np.diff(F_face, axis=1)               # shape (3, N)
        source = self.model.source(uu=uu, gamma=gamma)

        return -dF / dx + source

    def _apply_boundary_flux(
        self,
        F_face: np.ndarray,
        uu: np.ndarray,
        gamma: np.ndarray,
    ) -> None:
        """Overwrite boundary face fluxes with BC-imposed values (in-place).

        Uses the same Dirichlet / Neumann convention as :class:`FlowSolver`.
        """
        x = self.model.x
        flux = self.model.flux(uu=uu, gamma=gamma)

        # ---------- outlet (right boundary, face index N) ----------
        outlet = self.outlet_boundary
        scale_out = [1, self.model.area[-1], uu[1][-1]]
        offset_out = [0, 0.5 * (uu[1][-1] ** 2.0) / uu[0][-1], 0]
        cats = [BoundaryCategory.MF, BoundaryCategory.TP, BoundaryCategory.TH]
        for k, cat in enumerate(cats):
            if BoundaryType.DI in outlet[cat]:
                F_face[k, -1] = (list(outlet[cat].values())[0]
                                 * scale_out[k] + offset_out[k])
            elif BoundaryType.NE in outlet[cat]:
                F_face[k, -1] = flux[k, -1] + (list(outlet[cat].values())[0] * (
                    x[-1] - x[-2]) * scale_out[k] + offset_out[k])

        # ---------- intake (left boundary, face index 0) ----------
        intake = self.intake_boundary
        scale_in = [1, self.model.area[0], uu[1][0]]
        offset_in = [0, 0.5 * (uu[1][0] ** 2.0) / uu[0][0], 0]
        for k, cat in enumerate(cats):
            if BoundaryType.DI in intake[cat]:
                F_face[k, 0] = (
                    list(intake[cat].values())[0] * scale_in[k] + offset_in[k]
                )
            elif BoundaryType.NE in intake[cat]:
                F_face[k, 0] = flux[k, 0] - (
                    list(intake[cat].values())[0] * (x[1] - x[0]) * scale_in[k]
                    + offset_in[k]
                )

    # ------------------------------------------------------------------
    # Public time-advancement method
    # ------------------------------------------------------------------

    def update(self, dt: float = 1e-6) -> None:
        """Advance the solution by one time step using SSP-RK2.

        The Shu-Osher SSP-RK2 (Heun) scheme reads::

            u* = u^n + dt * L(u^n)
            u^{n+1} = 0.5 * u^n + 0.5 * (u* + dt * L(u*))

        which preserves total-variation diminishing properties when
        combined with a TVD spatial operator.

        Parameters
        ----------
        dt:
            Time-step size (seconds).  The caller is responsible for
            choosing a CFL-stable value.
        """
        gamma = np.asarray(self.gamma)

        # Stage 1
        rhs1 = self._residual(self.uu, gamma)
        uu_star = self.uu + dt * rhs1

        # Stage 2
        rhs2 = self._residual(uu_star, gamma)
        self.uu = 0.5 * self.uu + 0.5 * (uu_star + dt * rhs2)
