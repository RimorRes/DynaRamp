import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, Projectile, Slider
from dynaramp.contact import RailProfile, ContactSolver
from dynaramp.simulation import LaunchSimulator, LaunchResult, Gravity, Thrust


def _guide_system():
    topo = dyn.TopologyHandler()
    length = 10
    h, b = 1, 2
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


@pytest.fixture(scope="module")
def field():
    system = _guide_system()
    modes = system.natural_modes(2, omega_max=200, search_res=2000)
    assert len(modes) >= 1
    return GuideModalField.from_elements(system, ["beam"], modes)


def _projectile():
    return Projectile(
        mass=50.0, inertia_com=np.diag([2.0, 30.0, 30.0]), com_o1=(0.5, 0.0, 0.0),
        sliders=[Slider((0.0, 0.0, 0.0), 0.0), Slider((1.5, 0.0, 0.0), 0.0)],
    )


def _solver(field, projectile, l_c=np.inf, clearance=1e-3, stiffness=1e5):
    profile = RailProfile(clearance, clearance, clearance, stiffness=stiffness,
                          restitution=0.5, friction=0.1)
    return ContactSolver(field, profile, projectile.sliders, l_c=l_c)


def test_launch_runs_and_records(field):
    proj = _projectile()
    solver = _solver(field, proj)
    sim = LaunchSimulator(field, proj, solver, forces=[Gravity(), Thrust(curve=lambda t: 2000.0)])
    x0 = np.array([3.0, 0.0015, 0.0, 0.0, 0.0, 0.0])  # small lateral offset -> sliders in contact
    res = sim.run(x0, np.zeros(6), dt=1e-4, t_max=2e-3)

    n = field.n_modes
    assert isinstance(res, LaunchResult)
    steps = res.t.shape[0]
    assert res.x.shape == (steps, 6)
    assert res.y.shape == (steps, 6)
    assert res.p.shape == (steps, n)
    assert res.contact_force.shape == (steps, 3)
    for arr in (res.x, res.y, res.p, res.contact_force, res.contact_moment):
        assert np.all(np.isfinite(arr))
    assert res.attitude.shape == (steps, 3)
    # thrust along the body axis drives the projectile forward along the guide
    assert res.x[-1, 0] > res.x[0, 0]


def test_free_fall_without_contact(field):
    # Huge clearance -> no contact. Under gravity alone the projectile falls in -y.
    proj = _projectile()
    solver = _solver(field, proj, clearance=1.0)
    sim = LaunchSimulator(field, proj, solver, forces=[Gravity()])
    res = sim.run(np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6), dt=1e-3, t_max=1e-2)
    assert res.x[-1, 1] < res.x[0, 1]                      # O1 dropped
    # near free-fall: dropped roughly 0.5 g t^2 (loose bound)
    drop = res.x[0, 1] - res.x[-1, 1]
    assert 0 < drop < 0.5 * 9.81 * (1e-2) ** 2 * 2


def test_terminates_when_all_sliders_exited(field):
    proj = _projectile()
    solver = _solver(field, proj, l_c=2.5)                 # both slider stations (3.0, 4.5) past 2.5
    sim = LaunchSimulator(field, proj, solver, forces=[Gravity()])
    res = sim.run(np.array([3.0, 0.0015, 0.0, 0.0, 0.0, 0.0]), np.zeros(6), dt=1e-4, t_max=1e-2)
    assert res.exited is True
    assert res.t.shape[0] == 1                             # stops on the first step


def test_runs_to_tmax_when_never_exiting(field):
    proj = _projectile()
    solver = _solver(field, proj, l_c=np.inf)
    sim = LaunchSimulator(field, proj, solver, forces=[Gravity()])
    res = sim.run(np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6), dt=1e-3, t_max=5e-3)
    assert res.exited is False
    assert res.t.shape[0] == 5                             # ceil(5e-3 / 1e-3)
