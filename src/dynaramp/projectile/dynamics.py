from __future__ import annotations
import logging

from dataclasses import dataclass

import numpy as np

from ..common.types import Vector, Matrix
from ..common.vecmath import skew_sym_mat
from .projectile import Projectile
from .kinematics import ProjectileKinematicState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectileEOM:
    """
    The projectile's equations of motion within the launch rail, projected in K_I.

    Section 3.3, Eqs. 43-48. The six coefficient blocks map the launch-vehicle modal
    acceleration ``p_ddot`` and the projectile quasi-acceleration ``y_dot`` to the
    external resultants:

        ``M_Tp p_ddot + M_Ty y_dot = h_T + sum_k I_q_O1,k``   (translational, Eq. 43)
        ``M_Rp p_ddot + M_Ry y_dot = h_R + sum_k I_m_O1,k``   (rotational,   Eq. 44)

    These are exactly the blocks that section 5 stacks under the launch-vehicle modal
    EOM to form the coupled system (Eq. 67). The external force and moment resultants on
    the right-hand side -- contact from section 4, plus gravity and thrust -- are
    supplied separately.

    Attributes
    ----------
    m_tp : Matrix
        ``3 x n`` translational modal inertia block.
    m_ty : Matrix
        ``3 x 6`` translational quasi-velocity inertia block.
    m_rp : Matrix
        ``3 x n`` rotational modal inertia block.
    m_ry : Matrix
        ``3 x 6`` rotational quasi-velocity inertia block.
    h_t : Vector
        Translational convective/inertial resultant.
    h_r : Vector
        Rotational convective/inertial resultant.
    """
    m_tp: Matrix
    m_ty: Matrix
    m_rp: Matrix
    m_ry: Matrix
    h_t: Vector
    h_r: Vector

    @classmethod
    def assemble(cls, kin: ProjectileKinematicState, projectile: Projectile) -> "ProjectileEOM":
        """
        Assemble the six blocks from a kinematic state and the projectile's inertia.

        Parameters
        ----------
        kin : ProjectileKinematicState
            The section 3.1-3.2 kinematics at the current instant.
        projectile : Projectile
            The projectile's inertial properties.

        Returns
        -------
        ProjectileEOM
            The assembled blocks.
        """
        m = projectile.mass
        r_o1c = kin.a_ib @ projectile.com_o1                       # I_r_O1C (Eq. 48 note)
        i_o1 = kin.a_ib @ projectile.inertia_o1 @ kin.a_ib.T       # I_I_O1 (body inertia about O1, in K_I)
        rc = skew_sym_mat(r_o1c)
        w = skew_sym_mat(kin.omega_ib)

        m_tp = m * (kin.l_to1 - rc @ kin.l_rp)                                      # Eq. 45
        m_ty = m * (kin.j_to1 - rc @ kin.j_ro1)                                     # Eq. 45
        h_t = m * (rc @ kin.zeta_ro1 - w @ w @ r_o1c - kin.zeta_to1)                # Eq. 46

        m_rp = i_o1 @ kin.l_rp + m * rc @ kin.l_to1                                 # Eq. 47
        m_ry = i_o1 @ kin.j_ro1 + m * rc @ kin.j_to1                                # Eq. 47
        h_r = -i_o1 @ kin.zeta_ro1 - m * rc @ kin.zeta_to1 - w @ i_o1 @ kin.omega_ib  # Eq. 48

        return cls(m_tp=m_tp, m_ty=m_ty, m_rp=m_rp, m_ry=m_ry, h_t=h_t, h_r=h_r)
