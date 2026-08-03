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
    The projectile's equations of motion within the launch rail (section 3.3, Eqs. 43-48),
    projected in K_I. The six coefficient blocks map the launch-vehicle modal acceleration
    p_ddot and the projectile quasi-acceleration y_dot to the external resultants:

        M_Tp p_ddot + M_Ty y_dot = h_T + sum_k I_q_O1,k      (translational, Eq. 43)
        M_Rp p_ddot + M_Ry y_dot = h_R + sum_k I_m_O1,k      (rotational,   Eq. 44)

    These are exactly the blocks that section 5 stacks under the launch-vehicle modal EOM
    to form the coupled system (Eq. 67). The external force/moment resultants on the RHS
    (contact from section 4, plus gravity/thrust) are supplied separately.
    """
    m_tp: Matrix   # 3xn
    m_ty: Matrix   # 3x6
    m_rp: Matrix   # 3xn
    m_ry: Matrix   # 3x6
    h_t: Vector    # 3
    h_r: Vector    # 3

    @classmethod
    def assemble(cls, kin: ProjectileKinematicState, projectile: Projectile) -> "ProjectileEOM":
        """
        Assemble the six blocks from a kinematic state (section 3.1-3.2) and the projectile's
        inertial properties.
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

        return cls(
            m_tp=m_tp.astype(np.float64),
            m_ty=m_ty.astype(np.float64),
            m_rp=m_rp.astype(np.float64),
            m_ry=m_ry.astype(np.float64),
            h_t=h_t.astype(np.float64),
            h_r=h_r.astype(np.float64),
        )
