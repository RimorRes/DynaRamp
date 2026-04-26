import numpy as np
from scipy.linalg import solve

def newmarkbeta_integrator(m, c, k, f_func, init_state, t_stop, dt, beta=0.25, gamma=0.5):
    # TODO: CFL number, adaptive time step size
    # Initializing problem
    q, q_dot = init_state  # initial state vector, generalized coordinates
    q_ddot = 0

    # Main loop, time stepping
    for t in np.arange(0, t_stop, dt):
        qp, qp_dot, qp_ddot = q, q_dot, q_ddot  # previous state

        # Pre-calculate the explicit terms of the Ax = B problem for efficiency
        a = 1 / (beta * dt ** 2) * m + gamma / (beta * dt) * c + k
        b = m @ ((1 / (2 * beta) - 1) * qp_ddot + (1 / (beta * dt)) * qp_dot + (1 / (beta * (dt ** 2))) * qp) + \
            c @ ((gamma / (2 * beta) - 1) * dt * qp_ddot + (gamma / beta - 1) * qp_dot + (gamma / (beta * dt)) * qp)

        # Nonlinear dynamics, loop step until convergence
        conv = False
        # Predictor
        pred_ddot = q_ddot  # Constant acceleration is assumed, could be 0
        q = qp + dt * qp_dot + (dt ** 2) / 2 * ((1 - 2 * beta) * qp_ddot + 2 * beta * pred_ddot)
        q_dot = qp_dot + dt * ((1 - gamma) * qp_ddot + gamma * pred_ddot)
        # Corrector loop
        while not conv:
            qe = q
            qe_dot = q_dot
            # Evaluate forces f(q, q_dot, t) using estimated state
            f = f_func(qe, qe_dot, t)
            # Aq = (B+f) solves for q
            q = solve(a, b+f)
            q_dot = gamma/(beta*dt) * (q - qp) + (1 - gamma/beta)*qp_dot + (1 - gamma/(2*beta))*dt*qp_ddot

            # Check convergence
            conv = np.linalg.norm(q - qe) < 1e-6 # TODO: add convergence criteria as an argument

        # Calculate the acceleration term after convergence for efficiency, f doesn't depend on it
        q_ddot = 1/(beta*(dt**2)) * (q - qp - dt*qp_dot + (beta - 1/2)*dt**2*qp_ddot)

        # TIME STEP END
        yield q, q_dot, q_ddot


