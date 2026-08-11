from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ..common.types import Vector, VectorLike
from ..common.vecmath import moment_about
from ..projectile.kinematics import ProjectileKinematicState
from ..projectile.projectile import Projectile

logger = logging.getLogger(__name__)

# Standard gravity [m/s^2].
G0 = 9.80665


@runtime_checkable
class ExternalForce(Protocol):
    """
    A non-contact load on the projectile.

    Given the current projectile kinematics, the projectile properties and the time, it
    returns the force and moment it applies at O1, both projected in the inertial frame
    K_I: ``(I_q_O1, I_m_O1)``.
    """

    def __call__(
            self,
            kin: ProjectileKinematicState,
            projectile: Projectile,
            t: float,
    ) -> Tuple[Vector, Vector]:
        ...


@dataclass
class Gravity:
    """
    Uniform gravity.

    The force ``m g`` acts at the center of mass, so it produces a moment about O1
    through the COM lever arm ``I_r_O1C = A_IB B_r_O1C``.

    Attributes
    ----------
    g : VectorLike
        Gravitational acceleration vector in K_I [m/s^2]. Defaults to +z, the "down"
        axis of the NED convention (+X forward, +Y right, +Z down). With the launcher
        upright (``A_IR = I``) this seats the shoes on the rail-groove floor.
    """
    g: VectorLike = (0.0, 0.0, G0)

    def __call__(self, kin: ProjectileKinematicState, projectile: Projectile, t: float) -> Tuple[Vector, Vector]:
        """
        Evaluate the gravitational load at O1.

        Parameters
        ----------
        kin : ProjectileKinematicState
            Current projectile kinematics.
        projectile : Projectile
            The projectile's inertial properties.
        t : float
            Current time [s]. Unused; gravity is time-invariant.

        Returns
        -------
        tuple of Vector
            ``(force, moment)`` at O1, in K_I.
        """
        g_i = np.asarray(self.g, dtype=np.float64)
        force = projectile.mass * g_i
        r_o1c = kin.a_ib @ projectile.com_o1                 # I_r_O1C
        return force, moment_about(r_o1c, force)


@dataclass
class Thrust:
    """
    Rocket motor thrust along a body-fixed axis, with a time-dependent magnitude.

    Attributes
    ----------
    curve : Callable[[float], float]
        Thrust magnitude ``T(t)`` [N] as a function of time: motor curve, ramp,
        burnout, and so on.
    application_point : VectorLike
        ``B_r_O1T``, the thrust application point relative to O1, in the body frame K_B.
    axis : VectorLike
        Thrust direction in the body frame K_B. Defaults to +x, toward the nose.
    """
    curve: Callable[[float], float]
    application_point: VectorLike = (0.0, 0.0, 0.0)
    axis: VectorLike = (1.0, 0.0, 0.0)

    def __call__(self, kin: ProjectileKinematicState, projectile: Projectile, t: float) -> Tuple[Vector, Vector]:
        """
        Evaluate the thrust load at O1.

        Parameters
        ----------
        kin : ProjectileKinematicState
            Current projectile kinematics.
        projectile : Projectile
            The projectile's inertial properties. Unused.
        t : float
            Current time [s], passed to the thrust curve.

        Returns
        -------
        tuple of Vector
            ``(force, moment)`` at O1, in K_I.
        """
        magnitude = float(self.curve(t))
        axis_i = kin.a_ib @ np.asarray(self.axis, dtype=np.float64)      # thrust direction in K_I
        force = magnitude * axis_i
        r_o1t = kin.a_ib @ np.asarray(self.application_point, dtype=np.float64)
        return force, moment_about(r_o1t, force)


def sum_external_forces(
        forces: Sequence[ExternalForce],
        kin: ProjectileKinematicState,
        projectile: Projectile,
        t: float,
) -> Tuple[Vector, Vector]:
    """
    Sum a collection of external loads into a single resultant at O1.

    Parameters
    ----------
    forces : Sequence[ExternalForce]
        The loads to sum.
    kin : ProjectileKinematicState
        Current projectile kinematics.
    projectile : Projectile
        The projectile's inertial properties.
    t : float
        Current time [s].

    Returns
    -------
    tuple of Vector
        ``(sum I_q_O1, sum I_m_O1)``, the total non-contact force and moment at O1,
        in K_I.
    """
    q = np.zeros(3, dtype=np.float64)
    m = np.zeros(3, dtype=np.float64)
    for f in forces:
        fq, fm = f(kin, projectile, t)
        q = q + np.asarray(fq, dtype=np.float64)
        m = m + np.asarray(fm, dtype=np.float64)
    return q, m
