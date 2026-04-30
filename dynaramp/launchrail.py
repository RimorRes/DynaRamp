from .beam import EulerBernoulliBeam
import numpy as np
from scipy.integrate import quad


class LaunchRail:

    def __init__(self, beam: EulerBernoulliBeam, n_modes: int, angle: float):
        self.beam = beam
        self.n_modes = n_modes
        self.angle = angle  # Rail inclination from horizontal, radians

        # Numpy mode shapes (for plotting, integration)
        _, self.modal_shapes = self.beam.modes(1, n_modes)
        # JAX-differentiable mode shapes (for NNR tangent stiffness)
        _, self.jax_modal_shapes = self.beam.jax_modes(1, n_modes)

        # Precompute ∫₀ᴸ φᵢ(x)dx — gravity projection onto each mode
        self.shape_ints = [
            quad(lambda x: phi(x), 0, self.beam.L)[0]
            for phi in self.modal_shapes
        ]

        self.M, self.C, self.K = self.beam.modal_matrices(n_modes)

    def displacement(self, x: float, eta: np.ndarray) -> float:
        """Transverse beam deflection at abscissa x given modal amplitudes eta."""
        return sum(eta[i] * self.modal_shapes[i](x) for i in range(self.n_modes))

    def jax_displacement(self, x, eta):
        """JAX-differentiable transverse beam deflection."""
        return sum(eta[i] * self.jax_modal_shapes[i](x) for i in range(self.n_modes))
