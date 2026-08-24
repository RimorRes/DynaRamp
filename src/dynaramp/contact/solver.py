from __future__ import annotations
import logging

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Sequence

import numpy as np

from ..common.types import Vector, Matrix
from ..projectile.kinematics import ProjectileKinematicState
from ..projectile.modal_field import GuideModalField
from ..projectile.projectile import Slider
from .profile import GuideProfile
from .detection import SliderContact, evaluate_slider

logger = logging.getLogger(__name__)

# Per-slider, per-face impact-onset velocity memory: {slider_index: {face_label: delta_dot_minus}}.
ContactMemory = Dict[int, Dict[str, float]]


@dataclass(frozen=True)
class GuideReaction:
    """
    Contact reaction on the guide from one slider, for the section 5 modal projection.

    Attributes
    ----------
    x_r_i : float
        Axial station of the contact cross-section.
    i_q_pi : Vector
        Reaction force on the guide at that station, in K_I.
    i_m_pi : Vector
        Reaction moment on the guide at that station, in K_I.
    """
    x_r_i: float
    i_q_pi: Vector
    i_m_pi: Vector


@dataclass(frozen=True)
class ContactResult:
    """
    Aggregated section 4 output at one instant.

    Attributes
    ----------
    sum_q_o1 : Vector
        Total contact force on the projectile at O1 -- the Sigma term of Eq. 69 that
        section 5 adds to ``h_T``.
    sum_m_o1 : Vector
        Total contact moment on the projectile at O1 -- the Sigma term of Eq. 70 that
        section 5 adds to ``h_R``.
    guide_reactions : List[GuideReaction]
        Per-slider reactions on the guide, which section 5 projects onto the
        launch-vehicle modes to build ``f_g``.
    memory : ContactMemory
        Updated impact-onset velocity memory, to carry to the next time step.
    slider_contacts : List[SliderContact]
        Per-slider detail, in slider order, for inspection and plotting.
    """
    sum_q_o1: Vector
    sum_m_o1: Vector
    guide_reactions: List[GuideReaction]
    memory: ContactMemory
    slider_contacts: List[SliderContact] = dc_field(default_factory=list)


class ContactSolver:
    """
    Runs the section 4 contact analysis over all sliders against one guide profile.

    Sums the resultants and threads the impact-velocity memory. The solver is stateless:
    :meth:`evaluate` takes the previous memory and returns the updated one, so the time
    integrator (section 5) owns the state.

    Parameters
    ----------
    field : GuideModalField
        The guide modal field.
    profile : GuideProfile | Sequence[GuideProfile]
        The guide cross-section contact model. A single profile applies to every slider.
        A sequence assigns one per slider, which is what a guide with more than one
        groove needs: each slider rides its own, related to the cross-section origin by
        its own datum (see :class:`dynaramp.contact.profile.OffsetProfile`).
    sliders : Sequence[Slider]
        The projectile's sliders, in order.
    a_ir : Matrix | None
        Rotation ``A_IR`` from K_R to K_I. ``A_IR`` is a property of the guide, so it
        lives on the modal field; this defaults to the field's own value to keep the
        mode-shape projection and the contact-cross-section geometry in the same frame.
        An explicit value here overrides it (advanced use).
    l_c : float | Sequence[float]
        Per-slider exit station: the axial station past which a slider disengages from
        the guide (Eq. 61, generalized). A scalar applies to all sliders -- the paper's
        single canister exit, giving sequential detachment. A sequence sets an
        independent exit per slider, so sliders can be made to detach simultaneously,
        e.g. a rail groove whose cross-section widens along its length, releasing rear
        and front sliders at once.
    station_bracket : float
        Bracket size for the contact-station root find.

    Raises
    ------
    ValueError
        If ``l_c`` or ``profile`` is a sequence whose length does not match the slider
        count.
    """

    def __init__(
            self,
            field: GuideModalField,
            profile: GuideProfile | Sequence[GuideProfile],
            sliders: Sequence[Slider],
            a_ir: Matrix | None = None,
            l_c: float | Sequence[float] = np.inf,
            station_bracket: float = 0.5,
    ):
        self.field = field
        self.sliders: List[Slider] = list(sliders)

        if isinstance(profile, GuideProfile):
            self.profiles: List[GuideProfile] = [profile] * len(self.sliders)
        else:
            self.profiles = list(profile)
            if len(self.profiles) != len(self.sliders):
                raise ValueError(
                    f"profile has {len(self.profiles)} entries but there are "
                    f"{len(self.sliders)} sliders."
                )
        self.a_ir = np.asarray(field.a_ir if a_ir is None else a_ir, dtype=np.float64)
        self.station_bracket = float(station_bracket)

        if isinstance(l_c, (int, float)):
            self.exit_stations: List[float] = [float(l_c)] * len(self.sliders)
        else:
            self.exit_stations = [float(v) for v in l_c]
            if len(self.exit_stations) != len(self.sliders):
                raise ValueError(
                    f"l_c has {len(self.exit_stations)} entries but there are "
                    f"{len(self.sliders)} sliders."
                )

    def evaluate(
            self,
            kin: ProjectileKinematicState,
            x_r: float,
            p: Vector,
            p_dot: Vector,
            memory: ContactMemory | None = None,
    ) -> ContactResult:
        """
        Evaluate all sliders at the current state.

        Parameters
        ----------
        kin : ProjectileKinematicState
            Section 3 kinematics at O1.
        x_r : float
            Axial coordinate of O1.
        p : Vector
            Launch-vehicle modal coordinates.
        p_dot : Vector
            Launch-vehicle modal rates.
        memory : ContactMemory | None
            The previous step's onset-velocity memory.

        Returns
        -------
        ContactResult
            The summed resultants, the per-slider detail, and the updated memory.
        """
        memory = memory or {}
        sum_q = np.zeros(3, dtype=np.float64)
        sum_m = np.zeros(3, dtype=np.float64)
        reactions: List[GuideReaction] = []
        contacts: List[SliderContact] = []
        new_memory: ContactMemory = {}

        for idx, slider in enumerate(self.sliders):
            prev = memory.get(idx, {})
            sc = evaluate_slider(
                kin, self.field, self.a_ir, p, p_dot, slider, self.profiles[idx],
                x_r=x_r, l_c=self.exit_stations[idx], impact_velocities=prev,
                station_bracket=self.station_bracket,
            )
            contacts.append(sc)
            if not sc.in_phase or not sc.surfaces:
                continue

            sum_q += sc.i_q_o1
            sum_m += sc.i_m_o1
            reactions.append(GuideReaction(sc.x_r_i, sc.i_q_pi, sc.i_m_pi))

            # Latch the impact-onset velocity: keep it for faces already in contact, set it
            # to the current penetration velocity for faces making first contact.
            new_memory[idx] = {
                label: prev.get(label, d_dot)
                for label, d_dot in sc.penetration_velocities.items()
            }

        return ContactResult(
            sum_q_o1=sum_q,
            sum_m_o1=sum_m,
            guide_reactions=reactions,
            memory=new_memory,
            slider_contacts=contacts,
        )
