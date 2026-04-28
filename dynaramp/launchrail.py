from beam import EulerBernoulliBeam
import numpy as np
from scipy.integrate import quad


class LaunchRail:

    def __init__(self, beam: EulerBernoulliBeam, n_modes: int, angle: float):
        self.beam = beam

        self.n_modes = n_modes  # Number of modes to consider
        self.modal_shapes = self.beam.modes(1, n_modes)[1]  # Mode shapes as functions of x
        self.amps = np.zeros(n_modes)
        # Precompute the integrals of the mode shapes for later use in modal projection
        self.shape_ints = [quad(lambda x: shape(x), 0, self.beam.L)[0] for shape in self.modal_shapes]
        # Modal mass, damping, and stiffness matrices
        self.M, self.C, self.K = self.beam.modal_matrices(n_modes)

        self.angle = angle  # Ramp angle in radians

    def displacement(self, x: float):
        # Displacement of the beam at position x
        disp = 0
        for i in range(self.n_modes):
            disp += self.amps[i] * self.modal_shapes[i](x)
        return disp
