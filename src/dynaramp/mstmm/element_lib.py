from __future__ import annotations
import logging

from typing import Dict, Tuple

import numpy as np

from .structs import DiscreteElement, ContinuousElement, MasslessMixin
from .krylov import krylov_normalized, sinc
from ..common.vecmath import I3, parallel_axis, skew_sym_mat, spatial_mass_matrix
from ..common.types import EntityID, Vector, VectorLike, Matrix

logger = logging.getLogger(__name__)

__all__ = [
    "JunctionNode",
    "SpatialElasticHinge",
    "LumpedMass",
    "RigidBody",
    "EulerBernoulliBeam",
]

# Shared, read-only identity used by the frequency-independent elements. Handing out the
# same frozen array avoids rebuilding a 12x12 identity on every transfer-matrix query.
_I12 = np.identity(12, dtype=np.float64)
_I12.setflags(write=False)


class JunctionNode(MasslessMixin, DiscreteElement):
    """
    A massless, perfectly rigid connection point.

    Transmits its state unchanged, so its transfer matrix is the identity. Used to
    branch a topology without introducing inertia or compliance.
    """

    def _u_local(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        return _I12


class SpatialElasticHinge(MasslessMixin, DiscreteElement):
    """
    A massless six-degree-of-freedom elastic connection.

    Three linear springs and three torsional springs, acting along the hinge's own axes.
    The transfer matrix is frequency-independent because the element carries no inertia.

    Parameters
    ----------
    e_id : EntityID
        Unique element ID.
    k : tuple of float
        Linear spring stiffnesses along the local x, y, z axes.
    k_rot : tuple of float
        Torsional spring stiffnesses about the local x, y, z axes.
    orientation : Matrix | None
        Optional 3x3 direction-cosine matrix (local -> global) giving the hinge's
        absolute orientation. The stiffness axes rotate with it.
    """

    def __init__(
            self,
            e_id: EntityID,
            k: Tuple[float, float, float],
            k_rot: Tuple[float, float, float],
            orientation: Matrix | None = None,
    ):
        super().__init__(e_id, orientation)

        k_mat = np.diag(-1.0 / np.asarray(k, dtype=np.float64))
        k_rot_mat = np.diag(1.0 / np.asarray(k_rot, dtype=np.float64))

        u_mat = np.zeros((12, 12), dtype=np.float64)
        u_mat[0:6, 0:6] = np.identity(6)
        u_mat[6:12, 6:12] = np.identity(6)
        u_mat[0:3, 9:12] = k_mat
        u_mat[3:6, 6:9] = k_rot_mat
        # Frozen: `u` hands this array straight to the caller rather than copying it.
        u_mat.setflags(write=False)
        self._u_mat = u_mat

    def _u_local(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        return self._u_mat


class LumpedMass(DiscreteElement):
    """
    A point mass with no rotational inertia.

    Parameters
    ----------
    e_id : EntityID
        Unique element ID.
    mass : float
        Mass of the point.
    orientation : Matrix | None
        Optional 3x3 direction-cosine matrix (local -> global).
    """

    def __init__(self, e_id: EntityID, mass: float, orientation: Matrix | None = None):
        super().__init__(e_id, orientation)

        self.mass = float(mass)
        self._m_mat = spatial_mass_matrix(self.mass)
        self._m_mat.setflags(write=False)

    def _u_local(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        u_mat = np.identity(12, dtype=np.float64)
        u_mat[9:12, 0:3] = (self.mass * omega ** 2) * I3
        return u_mat

    @property
    def _m_param_mat(self) -> Matrix:
        return self._m_mat


class RigidBody(DiscreteElement):
    """
    A rigid body with arbitrary mass and inertia, and any number of ports.

    Parameters
    ----------
    e_id : EntityID
        Unique element ID.
    mass : float
        Mass of the body.
    inertia : Matrix
        3x3 inertia tensor about the center of mass, in the body frame.
    com : VectorLike
        Center-of-mass position relative to the body's origin, in the body frame.
    orientation : Matrix | None
        Optional 3x3 direction-cosine matrix (local -> global) giving the body's
        absolute orientation. The local port geometry, ``com`` and ``inertia`` are all
        interpreted in the body frame it defines.
    """

    MAX_INPUTS = None
    MAX_OUTPUTS = None

    def __init__(
            self,
            e_id: EntityID,
            mass: float,
            inertia: Matrix,
            com: VectorLike = (0, 0, 0),
            orientation: Matrix | None = None,
    ):
        super().__init__(e_id, orientation)

        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.com_pos = np.asarray(com, dtype=np.float64)
        self._m_mat = spatial_mass_matrix(self.mass, self.inertia)
        self._m_mat.setflags(write=False)

    def _u_local(self, input_pos: VectorLike, output_pos: VectorLike, omega: float) -> Matrix:
        input_pos_arr = np.asarray(input_pos, dtype=np.float64)
        output_pos_arr = np.asarray(output_pos, dtype=np.float64)
        # The vector FROM the input TO the output
        r_io = output_pos_arr - input_pos_arr
        # The vector FROM the input TO the center of mass
        r_ic = self.com_pos - input_pos_arr

        # Inertia about the input port.
        j = parallel_axis(self.inertia, self.mass, r_ic)

        l_io = skew_sym_mat(r_io)
        l_ic = skew_sym_mat(r_ic)
        l_co = l_io - l_ic  # Mathematically equivalent to skew(r_io - r_ic)

        w2 = omega ** 2
        mw2 = self.mass * w2

        u_mat = np.zeros((12, 12), dtype=np.float64)
        u_mat[0:3, 0:3] = I3
        u_mat[0:3, 3:6] = -l_io
        u_mat[3:6, 3:6] = I3
        u_mat[6:9, 0:3] = mw2 * l_co
        u_mat[6:9, 3:6] = -w2 * (self.mass * l_io @ l_ic + j)
        u_mat[6:9, 6:9] = I3
        u_mat[6:9, 9:12] = l_io
        u_mat[9:12, 0:3] = mw2 * I3
        u_mat[9:12, 3:6] = -mw2 * l_ic
        u_mat[9:12, 9:12] = I3
        return u_mat

    @property
    def _m_param_mat(self) -> Matrix:
        return self._m_mat


class EulerBernoulliBeam(ContinuousElement):
    """
    A uniform Euler-Bernoulli beam with axial, torsional and biaxial bending compliance.

    Parameters
    ----------
    e_id : EntityID
        Unique element ID.
    length : float
        Axial length of the beam.
    density : float
        Material mass density rho.
    youngs_mod : float
        Young's modulus E.
    shear_mod : float
        Shear modulus G.
    area : float
        Cross-sectional area A.
    i_y : float
        Second moment of area about the local y axis, governing bending in the x-z plane.
    i_z : float
        Second moment of area about the local z axis, governing bending in the x-y plane.
    orientation : Matrix | None
        Optional 3x3 direction-cosine matrix (local -> global).

    Notes
    -----
    The transfer matrix is assembled from the *normalized* Krylov-Duncan group (see
    :mod:`dynaramp.mstmm.krylov`), so no wave number is ever divided by. The matrix is
    therefore valid at ``omega = 0``, where it reduces exactly to the static transfer
    matrix of the beam, and it retains full precision at small argument, where the
    closed-form Krylov functions lose roughly eight digits to cancellation.
    """

    # TODO: Add support for non-uniform beams (e.g. tapered, variable cross-section)

    # Wave numbers depend only on frequency, so they are memoized per omega. The cap keeps
    # a long eigenfrequency sweep -- which visits each frequency exactly once and never
    # hits the cache -- from retaining an unbounded number of entries.
    _WAVE_CACHE_CAP = 4096

    def __init__(
            self,
            e_id: EntityID,
            length: float,
            density: float,
            youngs_mod: float,
            shear_mod: float,
            area: float,
            i_y: float,
            i_z: float,
            orientation: Matrix | None = None,
    ):
        super().__init__(e_id, orientation)

        self.length = float(length)
        self.rho = float(density)
        self.e = float(youngs_mod)
        self.g = float(shear_mod)
        self.a = float(area)  # Cross-sectional area
        self.iy = float(i_y)  # Second moment of area about y-axis
        self.iz = float(i_z)  # Second moment of area about z-axis
        self.jp = self.iy + self.iz  # Polar moment, by the perpendicular axis theorem
        self.mu = self.rho * self.a  # Linear mass density

        # Bending, axial and torsional rigidities, hoisted out of the hot path.
        self._e_iy = self.e * self.iy
        self._e_iz = self.e * self.iz
        self._e_a = self.e * self.a
        self._g_jp = self.g * self.jp
        self._rho_jp = self.rho * self.jp

        self._wave_cache: Dict[float, Tuple[float, float, float, float]] = {}

    def _wave_numbers(self, omega: float) -> Tuple[float, float, float, float]:
        """
        The four wave numbers at a given frequency, memoized.

        Parameters
        ----------
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        tuple of float
            ``(beta_x, lam_y, lam_z, gam_theta_x)``: the axial, the two bending and the
            torsional wave numbers. All vanish at ``omega = 0``.
        """
        cached = self._wave_cache.get(omega)
        if cached is not None:
            return cached

        w2 = omega ** 2
        mu_w2 = self.mu * w2
        wave = (
            float(np.sqrt(mu_w2 / self._e_a)),
            float(np.power(mu_w2 / self._e_iz, 0.25)),
            float(np.power(mu_w2 / self._e_iy, 0.25)),
            float(np.sqrt(self.rho * w2 / self.g)),
        )

        if len(self._wave_cache) >= self._WAVE_CACHE_CAP:
            self._wave_cache.clear()
        self._wave_cache[omega] = wave
        return wave

    def _quad_order(self, omega: float, span_length: float) -> int:
        """
        Quadrature order scaled to the number of half-waves across the span.

        A high mode oscillates several times along the beam, and a fixed low order
        would under-resolve it. Roughly six points per half-wave is comfortably
        beyond the point where Gauss-Legendre has converged for this integrand.

        Parameters
        ----------
        omega : float
            Frequency of the modes being paired [rad/s].
        span_length : float
            Length of the interval being integrated over.

        Returns
        -------
        int
            The quadrature order, never below ``QUAD_ORDER`` nor above 512.
        """
        _, lam_y, lam_z, _ = self._wave_numbers(omega)
        half_waves = max(lam_y, lam_z) * span_length / np.pi
        return int(np.clip(np.ceil(6.0 * half_waves), self.QUAD_ORDER, 512))

    def _u_local(self, input_pos: VectorLike, output_pos: VectorLike, omega: float) -> Matrix:
        input_pos_arr = np.asarray(input_pos, dtype=np.float64)
        output_pos_arr = np.asarray(output_pos, dtype=np.float64)
        x = float((output_pos_arr - input_pos_arr)[0])

        beta_x, lam_y, lam_z, gam_theta_x = self._wave_numbers(omega)

        # Normalized Krylov groups for the two bending planes. Each call costs one set of
        # four transcendental evaluations and supplies every bending entry below; the
        # previous formulation re-derived them eighteen times per matrix.
        s_y, t1_y, u2_y, v3_y = krylov_normalized(lam_y * x)
        s_z, t1_z, u2_z, v3_z = krylov_normalized(lam_z * x)

        # Axial and torsional cardinal sines, playing the same wave-number-canceling role.
        sinc_ax = sinc(beta_x * x)
        sinc_to = sinc(gam_theta_x * x)

        # mu * omega^2 == E*Iz*lam_y^4 == E*Iy*lam_z^4 == E*A*beta_x^2, which is what lets
        # every inertial entry below be written without a wave number.
        mw2 = self.mu * omega ** 2
        x2, x3 = x * x, x * x * x
        e_iy, e_iz = self._e_iy, self._e_iz

        u_mat = np.zeros((12, 12), dtype=np.float64)

        # --- axial (x) ---
        u_mat[0, 0] = u_mat[9, 9] = np.cos(beta_x * x)
        u_mat[0, 9] = -x * sinc_ax / self._e_a
        u_mat[9, 0] = mw2 * x * sinc_ax

        # --- torsion (about x) ---
        u_mat[3, 3] = u_mat[6, 6] = np.cos(gam_theta_x * x)
        u_mat[3, 6] = x * sinc_to / self._g_jp
        u_mat[6, 3] = -self._rho_jp * omega ** 2 * x * sinc_to

        # --- bending in the x-y plane (governed by I_z) ---
        u_mat[1, 1] = u_mat[5, 5] = u_mat[8, 8] = u_mat[10, 10] = s_y
        u_mat[1, 5] = u_mat[8, 10] = x * t1_y
        u_mat[1, 8] = u_mat[5, 10] = x2 * u2_y / e_iz
        u_mat[1, 10] = x3 * v3_y / e_iz
        u_mat[5, 8] = x * t1_y / e_iz
        u_mat[5, 1] = u_mat[10, 8] = mw2 * x3 * v3_y / e_iz
        u_mat[8, 1] = u_mat[10, 5] = mw2 * x2 * u2_y
        u_mat[8, 5] = mw2 * x3 * v3_y
        u_mat[10, 1] = mw2 * x * t1_y

        # --- bending in the x-z plane (governed by I_y) ---
        u_mat[2, 2] = u_mat[4, 4] = u_mat[7, 7] = u_mat[11, 11] = s_z
        u_mat[2, 4] = u_mat[7, 11] = -x * t1_z
        u_mat[2, 7] = u_mat[4, 11] = -x2 * u2_z / e_iy
        u_mat[2, 11] = x3 * v3_z / e_iy
        u_mat[4, 7] = x * t1_z / e_iy
        u_mat[4, 2] = u_mat[11, 7] = -mw2 * x3 * v3_z / e_iy
        u_mat[7, 2] = u_mat[11, 4] = -mw2 * x2 * u2_z
        u_mat[7, 4] = mw2 * x3 * v3_z
        u_mat[11, 2] = mw2 * x * t1_z

        return u_mat

    @property
    def _m_bar_param_mat(self) -> Matrix:
        return spatial_mass_matrix(self.mu, np.diag([self._rho_jp, 0.0, 0.0]))
