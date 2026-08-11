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
    One active contact between a slider and a guide face, expressed in K_Pi.

    Returned by a :class:`GuideProfile` when the slider penetrates that face.

    Attributes
    ----------
    label : str
        Identifier of the face, e.g. ``"side_+y"``, ``"bottom"``, ``"top_lip"``.
    penetration : float
        ``delta``, the penetration depth into the face (> 0).
    normal : Vector
        Unit normal in K_Pi pointing in the direction the contact force pushes the
        slider -- away from the face, back toward the clearance interior.
    stiffness : float
        ``K``, the generalized contact stiffness for this face. See
        :mod:`dynaramp.contact.contact_model`.
    exponent : float
        ``n``, the force-penetration exponent: 1.5 for a Hertzian sphere-plane contact,
        1.0 for a flat conformal one.
    restitution : float
        ``e``, coefficient of restitution for this face.
    friction : float
        ``mu``, Coulomb friction coefficient for this face.
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
    Cross-section contact model of a guide.

    Given a slider's reference point in the cross-section frame K_Pi, it returns the
    active contact faces.

    Notes
    -----
    Coordinate convention (NED): x is the axial (forward) axis normal to the
    cross-section, y the lateral (right) axis, z the vertical (down) axis. The
    cross-section plane is therefore y-z, with +z pointing "down" so that gravity seats
    a shoe onto the +z (floor) face.
    """

    @abstractmethod
    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        """
        Active contacts for a slider at a given position in the cross-section.

        Parameters
        ----------
        r_vi : Vector
            The slider's reference point -- the ball center Vi, or a shoe reference
            point -- in K_Pi.
        radius : float
            Slider ball-head radius. Used by sphere-based profiles, ignored by
            conformal ones.
        station : float
            Axial station ``x_R,i`` of the contact cross-section. Used by
            station-varying profiles, such as a groove that widens along the guide, and
            ignored by uniform ones.

        Returns
        -------
        List[SurfaceContact]
            One entry per face with positive penetration; empty when the slider is
            floating free within the clearance.
        """
        raise NotImplementedError


class UniformProfile(GuideProfile, ABC):
    """
    Base class for a guide whose cross-section and contact parameters are constant.

    Holds the material and contact-law parameters shared by every face of the section,
    and builds the :class:`SurfaceContact` records from them. A concrete subclass then
    only has to express its own geometry: which faces exist, and how far the slider has
    penetrated each. This is the same template-method split that
    :class:`dynaramp.mstmm.structs.Element` uses between ``u`` and ``_u_local``.

    Parameters
    ----------
    stiffness : float
        ``K``, the generalized contact stiffness applied to every face.
    restitution : float
        ``e``, the coefficient of restitution, in [0, 1].
    friction : float
        ``mu``, the Coulomb friction coefficient.
    exponent : float
        ``n``, the force-penetration exponent.
    """

    def __init__(
            self,
            stiffness: float,
            restitution: float,
            friction: float,
            exponent: float,
    ):
        self.stiffness = float(stiffness)
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.exponent = float(exponent)

    def _surface(self, label: str, penetration: float, normal: VectorLike) -> SurfaceContact:
        """
        Build a contact record for one face, stamped with the profile's parameters.

        Parameters
        ----------
        label : str
            Identifier of the face.
        penetration : float
            Penetration depth into that face.
        normal : VectorLike
            Unit normal in K_Pi, pointing back toward the clearance interior.

        Returns
        -------
        SurfaceContact
            The contact record.
        """
        return SurfaceContact(
            label=label,
            penetration=float(penetration),
            normal=np.asarray(normal, dtype=np.float64),
            stiffness=self.stiffness,
            exponent=self.exponent,
            restitution=self.restitution,
            friction=self.friction,
        )


class RailProfile(UniformProfile):
    """
    An open rail groove engaged by a flat T-profile shoe.

    A rectangular clearance with up to four flat faces: two side walls (+/- y, lateral)
    and, along the vertical z axis of the NED cross-section, a bottom floor (+z, "down")
    and a top lip (-z, "up"). The clearances are the half-play of the shoe reference
    point about the groove center (the K_Pi origin), so they already fold in the shoe's
    own dimensions.

    Parameters
    ----------
    lateral_clearance : float
        Half-play along +/- y.
    bottom_clearance : float
        Play down to the groove floor (+z).
    top_clearance : float
        Play up to the rail lips (-z).
    stiffness : float
        ``K``, the contact stiffness. See
        :func:`dynaramp.contact.contact_model.linear_contact_stiffness`.
    restitution : float
        ``e``, the coefficient of restitution.
    friction : float
        ``mu``, the Coulomb friction coefficient.
    exponent : float
        ``n``, the force-penetration exponent. Contact is conformal (flat), so the
        intended value is 1.

    Notes
    -----
    The slider radius is ignored: a flat shoe's geometry lives in the clearances.
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
        super().__init__(stiffness, restitution, friction, exponent)
        self.lateral_clearance = float(lateral_clearance)
        self.bottom_clearance = float(bottom_clearance)
        self.top_clearance = float(top_clearance)

    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        """
        Active contacts against the four faces of the rectangular groove.

        Parameters
        ----------
        r_vi : Vector
            The shoe reference point in K_Pi.
        radius : float
            Ignored.
        station : float
            Ignored.

        Returns
        -------
        List[SurfaceContact]
            One entry per penetrated face.
        """
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


class CanisterProfile(UniformProfile):
    """
    The launch canister groove of the paper (Fig. 4), engaged by a spherical slider.

    The canister carries its grooves as horizontal slots machined into the two side
    walls (Fig. 10b), so the cross-section is anisotropic and the two clearances are
    *not* interchangeable. In the NED cross-section frame K_Pi (``e^Pi_y`` lateral/right,
    ``e^Pi_z`` vertical/down):

    * the slot's flanks trap the slider VERTICALLY, from above and below, giving a
      two-sided pair of walls at ``+/- clearance_z`` along ``e^Pi_z`` (Eqs. 57-58);
    * the slot's far end limits the slider LATERALLY -- the "bottom" of the groove in
      the sense of its deepest point, not its floor -- as a single wall at
      ``+clearance_y`` along ``e^Pi_y`` (Eqs. 59-60).

    Parameters
    ----------
    clearance_y : float
        The lateral (depth) gap.
    clearance_z : float
        The vertical (flank) gap.
    stiffness : float
        ``K``, the contact stiffness. See
        :func:`dynaramp.contact.contact_model.hertz_stiffness`.
    restitution : float
        ``e``, the coefficient of restitution.
    friction : float
        ``mu``, the Coulomb friction coefficient.
    exponent : float
        ``n``, the force-penetration exponent. Hertzian sphere-plane contact gives 1.5.

    Notes
    -----
    The paper's own frame is (+x forward, +y up, +z right), the mirror of NED in this
    respect: its ``c_y`` is the vertical clearance and belongs here in ``clearance_z``,
    and its ``c_z`` is the lateral one and belongs in ``clearance_y``. Getting this
    backwards is silent -- the simulation runs and returns plausible nonsense.

    The clearances are ball-surface-to-wall gaps, as drawn in Fig. 4c, so the ball
    radius is folded into them; the radius enters only the Hertzian stiffness, not this
    penetration geometry.
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
        super().__init__(stiffness, restitution, friction, exponent)
        self.clearance_y = float(clearance_y)
        self.clearance_z = float(clearance_z)

    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        """
        Active contacts against the groove flanks and its deep end.

        Parameters
        ----------
        r_vi : Vector
            The ball center Vi in K_Pi.
        radius : float
            Ignored: the radius is already folded into the clearances.
        station : float
            Ignored.

        Returns
        -------
        List[SurfaceContact]
            One entry per penetrated face.
        """
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
    A guide whose cross-section changes along its length.

    Delegates to a profile chosen per axial station. This models, for example, a rail
    groove that widens from the rear to the front so that sliders progressively -- or
    simultaneously -- disengage as the projectile advances.

    Parameters
    ----------
    profile_at : Callable[[float], GuideProfile]
        Maps an axial station ``x_R`` to the guide profile in force there, typically a
        :class:`RailProfile` or :class:`CanisterProfile` with station-dependent
        clearances.

    Notes
    -----
    The returned :class:`SurfaceContact` objects are unchanged, so the contact solver's
    output API is identical to the uniform-profile case.
    """

    def __init__(self, profile_at: Callable[[float], GuideProfile]):
        self.profile_at = profile_at

    def contacts(self, r_vi: Vector, radius: float = 0.0, station: float = 0.0) -> List[SurfaceContact]:
        """
        Delegate to the profile in force at the given station.

        Parameters
        ----------
        r_vi : Vector
            The slider's reference point in K_Pi.
        radius : float
            Slider ball-head radius, forwarded to the delegate.
        station : float
            Axial station selecting the delegate.

        Returns
        -------
        List[SurfaceContact]
            Whatever the delegate returns.
        """
        return self.profile_at(station).contacts(r_vi, radius=radius, station=station)
