from __future__ import annotations
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from ..common.types import Vector, VectorLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurfaceContact:
    """
    One active contact between a slider and a guide face, expressed in the cross-section
    frame K_Pi. Returned by a `GuideProfile` when the slider penetrates that face.

    Attributes
    ----------
    label : str
        Identifier of the face (e.g. "side_+y", "bottom", "top_lip").
    penetration : float
        delta, the penetration depth into the face (> 0).
    normal : Vector
        Unit normal in K_Pi pointing in the direction the contact force pushes the slider
        (away from the face, back toward the clearance interior).
    stiffness : float
        K, the generalized/contact stiffness for this face (see contact_model helpers).
    exponent : float
        n, the force-penetration exponent (1.5 Hertzian sphere-plane, 1.0 flat/conformal).
    restitution : float
        e, coefficient of restitution for this face.
    friction : float
        mu, Coulomb friction coefficient for this face.
    """
    label: str
    penetration: float
    normal: Vector
    stiffness: float
    exponent: float
    restitution: float
    friction: float


class GuideProfile(ABC):
    """
    Cross-section contact model of a guide. Given a slider's reference point in the
    cross-section frame K_Pi, it returns the active contact faces.

    Coordinate convention (NED): x = the axial (forward) axis normal to the cross-section,
    y = the lateral (right) axis, z = the vertical (down) axis. The cross-section plane is
    therefore y-z, with +z pointing "down" so gravity seats a shoe onto the +z (floor) face.
    """

    @abstractmethod
    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        """
        Active contacts for a slider whose reference point (the ball center Vi, or a shoe
        reference point) is at `r_vi` in K_Pi. `radius` is the slider ball-head radius,
        used by sphere-based profiles and ignored by conformal ones. `station` is the axial
        station x_R,i of the contact cross-section, used by station-varying profiles (e.g.
        a groove that widens along the guide) and ignored by uniform ones.
        """
        raise NotImplementedError


class RailProfile(GuideProfile):
    """
    An open rail groove engaged by a (flat, T-profile) shoe: a rectangular clearance with
    up to four flat faces -- two side walls (+/- y, lateral) and, along the vertical z axis
    of the NED cross-section, a bottom floor (+z, "down") and a top lip (-z, "up"). The
    clearances are the half-play of the shoe reference point about the groove center
    (K_Pi origin), i.e. they already fold in the shoe's own dimensions.

    Contact is conformal (flat), so the intended force model is exponent n = 1 with a
    linear stiffness (see `contact_model.linear_contact_stiffness`). The slider radius is
    ignored.
    """

    def __init__(
            self,
            lateral_clearance: float,
            bottom_clearance: float,
            top_clearance: float,
            stiffness: float,
            restitution: float,
            friction: float,
            exponent: float = 1.0,
    ):
        self.lateral_clearance = float(lateral_clearance)
        self.bottom_clearance = float(bottom_clearance)
        self.top_clearance = float(top_clearance)
        self.stiffness = float(stiffness)
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.exponent = float(exponent)

    def _surface(self, label: str, penetration: float, normal: VectorLike) -> SurfaceContact:
        return SurfaceContact(
            label=label,
            penetration=float(penetration),
            normal=np.asarray(normal, dtype=np.float64),
            stiffness=self.stiffness,
            exponent=self.exponent,
            restitution=self.restitution,
            friction=self.friction,
        )

    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        y, z = float(r_vi[1]), float(r_vi[2])
        out: List[SurfaceContact] = []

        # Side walls (+/- y, lateral): contact when the lateral offset exceeds the clearance.
        d = y - self.lateral_clearance
        if d > 0.0:
            out.append(self._surface("side_+y", d, [0.0, -1.0, 0.0]))
        d = -y - self.lateral_clearance
        if d > 0.0:
            out.append(self._surface("side_-y", d, [0.0, 1.0, 0.0]))

        # Bottom floor (+z, "down" in NED): the gravity-seated face; contact when the shoe
        # drops (+z) past it. The floor reacts back up (-z).
        d = z - self.bottom_clearance
        if d > 0.0:
            out.append(self._surface("bottom", d, [0.0, 0.0, -1.0]))

        # Top lip (-z, "up"): contact when the shoe lifts against the rail lips; reacts down (+z).
        d = -z - self.top_clearance
        if d > 0.0:
            out.append(self._surface("top_lip", d, [0.0, 0.0, 1.0]))

        return out


class CanisterProfile(GuideProfile):
    """
    The paper's launch canister groove (Fig. 4), engaged by a spherical slider (ball head).

    The canister carries its grooves as horizontal slots machined into the two side walls
    (Fig. 10b), so the cross-section is anisotropic and the two clearances are *not*
    interchangeable. In the NED cross-section frame K_Pi (e^Pi_y lateral/right,
    e^Pi_z vertical/down):

      * the slot's flanks trap the slider VERTICALLY, from above and below, giving a
        two-sided pair of walls at +/- clearance_z along e^Pi_z (Eqs. 57-58);
      * the slot's far end limits the slider LATERALLY -- the "bottom" of the groove in
        the sense of its deepest point, not its floor -- as a single wall at
        +clearance_y along e^Pi_y (Eqs. 59-60).

    So `clearance_z` is the vertical (flank) gap and `clearance_y` the lateral (depth)
    gap. Note the paper's own frame is (+x forward, +y up, +z right), the mirror of NED
    in this respect: its c_y is the vertical clearance and belongs here in `clearance_z`,
    and its c_z is the lateral one and belongs in `clearance_y`. Getting this backwards
    is silent -- the simulation runs and returns plausible nonsense.

    The clearances are ball-surface-to-wall gaps (as drawn in Fig. 4c), so the ball radius
    is folded into them; the radius enters only the Hertzian stiffness (exponent n = 1.5,
    `contact_model.hertz_stiffness`), not this penetration geometry.
    """

    def __init__(
            self,
            clearance_y: float,
            clearance_z: float,
            stiffness: float,
            restitution: float,
            friction: float,
            exponent: float = 1.5,
    ):
        self.clearance_y = float(clearance_y)
        self.clearance_z = float(clearance_z)
        self.stiffness = float(stiffness)
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.exponent = float(exponent)

    def _surface(self, label: str, penetration: float, normal: VectorLike) -> SurfaceContact:
        return SurfaceContact(
            label=label,
            penetration=float(penetration),
            normal=np.asarray(normal, dtype=np.float64),
            stiffness=self.stiffness,
            exponent=self.exponent,
            restitution=self.restitution,
            friction=self.friction,
        )

    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        y, z = float(r_vi[1]), float(r_vi[2])
        out: List[SurfaceContact] = []

        # Groove flanks at +/- clearance_z, trapping the slider vertically (Eqs. 57-58).
        d_z = abs(z) - self.clearance_z
        if d_z > 0.0:
            out.append(self._surface("flank", d_z, [0.0, 0.0, -float(np.sign(z))]))

        # Groove bottom -- the deep end of the slot -- at +clearance_y (Eqs. 59-60):
        # contact when the slider is pushed laterally into it; the wall reacts back in -y.
        d_y = y - self.clearance_y
        if d_y > 0.0:
            out.append(self._surface("bottom", d_y, [0.0, -1.0, 0.0]))

        return out


class StationVaryingProfile(GuideProfile):
    """
    A guide whose cross-section changes along its length: it delegates to a profile chosen
    per axial station. This models, e.g., a rail groove that widens from the rear to the
    front so sliders progressively (or simultaneously) disengage as the projectile advances.

    Parameters
    ----------
    profile_at : Callable[[float], GuideProfile]
        Maps an axial station x_R to the guide profile in force there (typically a
        RailProfile / CanisterProfile with station-dependent clearances).

    The returned `SurfaceContact` objects are unchanged, so the contact solver's output API
    is identical to the uniform-profile case.
    """

    def __init__(self, profile_at: Callable[[float], GuideProfile]):
        self.profile_at = profile_at

    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        return self.profile_at(station).contacts(r_vi, radius=radius, station=station)
