import numpy as np

from beam import EulerBernoulliBeam
from physics import nnr_solver
from launchrail import LaunchRail
from vehicle import RigidRocket2D, SimpleMotor, Shoe

class System:

    def __init__(self, ramp: LaunchRail, vehicle: RigidRocket2D):
        self.ramp = ramp
        self.vehicle = vehicle

        n = self.ramp.n_modes + 3
        self.M = np.zeros((n, n))
        self.M[0,0] = self.vehicle.mass
        self.M[1,1] = self.vehicle.mass
        self.M[2,2] = self.vehicle.J
        self.M[3:, 3:] = self.ramp.M

        self.C = np.zeros_like(self.M)
        self.C[3:, 3:] = self.ramp.C

    def external_forces(self, q, q_dot, t):
        f_ext = np.zeros(3+self.ramp.n_modes)
        # Modal projection of the external forces
        # f_s
        f_ext[0] = self.vehicle.thrust_vec[0] - self.vehicle.mass*9.81*np.sin(self.ramp.angle)
        for shoe in self.vehicle.shoes:
            beam_disp = self.ramp.displacement(shoe.contact_loc)
            f_ext[0] += - shoe.f_coef * np.abs(shoe.force_norm(beam_disp)) * np.sign(q_dot[0])

        # f_y
        f_ext[1] = self.vehicle.thrust_vec[1] - self.vehicle.mass*9.81*np.cos(self.ramp.angle)
        # f_theta
        f_ext[2] = 0
        # f_mode_amps
        f_ext[3:] = -self.ramp.beam.mu * 9.81 * np.cos(self.ramp.angle) * np.array(self.ramp.shape_ints)
        return f_ext

    def internal_forces(self, q):
        f_ints = np.zeros_like(q)
        if len(q.shape) == 1:
            loops = 1
        else:
            loops = q.shape[1]
        for j in range(loops):
            pos = np.array([q[0][j], q[1][j], 0])
            print(pos)
            self.vehicle.update(pos, q[2][j])

            f_int = np.zeros(3+self.ramp.n_modes)
            # Modal projection of the internal forces
            # f_s
            f_int[0] = 0

            for shoe in self.vehicle.shoes:
                beam_disp = self.ramp.displacement(shoe.contact_loc)
                nk = shoe.force_norm(beam_disp)
                # f_y
                f_int[1] += nk
                # f_theta
                f_int[2] += shoe.dk * nk
                # f_mode_amps
                for i in range(self.ramp.n_modes):
                    f_int[3+i] += - shoe.dk * nk * self.ramp.modal_shapes[i](shoe.contact_loc)
                f_int[3:] += self.ramp.K @ q[3:]
            f_ints.append(f_int)
        return f_ints


if __name__ == "__main__":
    def make_beam():
        # Steel rectangular beam
        b = 0.5
        h = 1
        A = b * h
        rho = 7850
        mu = rho * A  # linear mass density, kg/m
        E = 210e9  # Elastic modulus, Pa
        I = b * h ** 3 / 12  # Second moment of area, m^4
        L = 20  # Length of the beam, m

        return EulerBernoulliBeam(mu, E, I, L)

    Beam = make_beam()
    Rail = LaunchRail(Beam, 4, np.pi / 4)

    # Based on Black Brandt X Rocket
    Motor = SimpleMotor(257e3, 280)
    mass = 2600
    r = 0.44/2
    h = 14.50
    inertia = 1/12 * mass * (3*r**2 + h**2)
    Rocket = RigidRocket2D(Motor, mass, inertia)
    Rocket.add_shoe(rel_pos=np.array([5, r]), friction_coef=0.5)
    Rocket.add_shoe(rel_pos=np.array([-5, r]), friction_coef=0.5)


    Sys = System(Rail, Rocket)
    q0 = np.zeros(3+Rail.n_modes)
    q0[:3] = [5, -(r+0.1), 0]
    q_dot0 = np.zeros(3+Rail.n_modes)

    for vals in nnr_solver(
        m=Sys.M,
        c=Sys.C,
        f_int_func=Sys.internal_forces,
        f_ext_func=Sys.external_forces,
        init_state=(q0, q_dot0),
        t_stop=1,
        dt=0.01,
    ):
        print(vals)
