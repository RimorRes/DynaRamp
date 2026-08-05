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
    cross-section frame K_Pi (origin on the guide central axis, x = axis normal, y/z the
    principal cross-section axes), it returns the active contact faces.
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
    up to four flat faces -- two side walls (+/- y), a bottom floor (-z) and a top lip (+z).
    The clearances are the half-play of the shoe reference point about the groove center
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

        # Side walls (+/- y): contact when lateral offset exceeds the lateral clearance.
        d = y - self.lateral_clearance
        if d > 0.0:
            out.append(self._surface("side_+y", d, [0.0, -1.0, 0.0]))
        d = -y - self.lateral_clearance
        if d > 0.0:
            out.append(self._surface("side_-y", d, [0.0, 1.0, 0.0]))

        # Bottom floor (-z): the gravity-seated face; contact when the shoe drops past it.
        d = -z - self.bottom_clearance
        if d > 0.0:
            out.append(self._surface("bottom", d, [0.0, 0.0, 1.0]))

        # Top lip (+z): contact when the shoe lifts against the rail lips.
        d = z - self.top_clearance
        if d > 0.0:
            out.append(self._surface("top_lip", d, [0.0, 0.0, -1.0]))

        return out


class CanisterProfile(GuideProfile):
    """
    The paper's launch canister groove (Fig. 4), engaged by a spherical slider (ball head).
    In the cross-section frame K_Pi, e^Pi_y is the vertical (side) axis and e^Pi_z the
    lateral axis into the groove depth. The groove has two side walls (+/- y, clearance
    clearance_y) and a bottom -- the deep radial wall at +z (clearance clearance_z) that the
    slider is pressed into; it is open toward the missile on the -z side.

    The clearances are ball-surface-to-wall gaps (as drawn in Fig. 4c), so the ball radius
    is folded into them; the radius enters only the Hertzian stiffness (exponent n = 1.5,
    `contact_model.hertz_stiffness`), not this penetration geometry. Contact occurs on the
    side when |y| > clearance_y (Eqs. 57-58) and on the bottom when z > clearance_z
    (Eqs. 59-60).
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

        # Side walls at +/- clearance_y (Eqs. 57-58).
        d_y = abs(y) - self.clearance_y
        if d_y > 0.0:
            out.append(self._surface("side", d_y, [0.0, -float(np.sign(y)), 0.0]))

        # Bottom = deep radial wall at +z (Eqs. 59-60): contact when the slider is pressed
        # past the bottom clearance; the wall reacts back in -z.
        d_z = z - self.clearance_z
        if d_z > 0.0:
            out.append(self._surface("bottom", d_z, [0.0, 0.0, -1.0]))

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
