import numpy as np
from scipy.linalg import solve
from scipy.differentiate import jacobian

def newmarkbeta_solver(m, c, k, f_func, init_state, t_stop, dt, beta=0.25, gamma=0.5):
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


def nnr_solver(m, c, f_int_func, f_ext_func, init_state, t_stop, dt, beta=0.25, gamma=0.5, conv_err=1e-6):
    # Newmark/Newton-Raphson ODE solver, for nonlinear dynamics.
    # TODO: CFL number, adaptive time step size
    # Initializing problem
    q, q_dot = init_state  # initial state vector, generalized coordinates
    # Evaluate initial acceleration using the equation of motion, TODO: F_EXT CHECK ARGUMENTS
    q_ddot = np.linalg.inv(m) @ (f_ext_func(q, q_dot, 0) - f_int_func(q) - c @ q_dot)

    # Main loop, time stepping
    for t in np.arange(0, t_stop, dt):
        qp, qp_dot, qp_ddot = q, q_dot, q_ddot  # previous state

        # Nonlinear dynamics, loop step until convergence

        # Explicit predictor
        qk_ddot = q_ddot  # Constant acceleration is assumed, could be 0
        qk_dot = qp_dot + dt * ((1 - gamma) * qp_ddot + gamma * qk_ddot)
        qk = qp + dt * qp_dot + (dt ** 2) / 2 * ((1 - 2 * beta) * qp_ddot + 2 * beta * qk_ddot)

        # Newton-Raphson corrector loop
        conv = False
        while not conv:

            # Evaluate forces using estimated state
            f_ext = f_ext_func(qk, qk_dot, t)  # TODO: CHECK ARGUMENTS
            f_int = f_int_func(qk)
            # Evaluate jacobian of internal forces
            print("qk:", qk)
            res = jacobian(f_int_func, qk)
            assert res.status == 0
            k_tangent = res.df  # Slope of internal forces, equivalent stiffness at qk

            residual = f_ext - f_int - m @ qk_ddot - c @ qk_dot

            m_coef = 1 / (beta * (dt ** 2))  # = delta_q_ddot / delta_q
            c_coef = gamma / (beta * dt)  # = delta_q_dot / delta_q
            k_hat = m_coef * m + c_coef * c + k_tangent  # Effective stiffness matrix
            delta_q = solve(k_hat, residual)

            # Update state
            qk += delta_q
            qk_dot += c_coef * delta_q
            qk_ddot += m_coef * delta_q

            conv = np.linalg.norm(delta_q) < conv_err

        # TIME STEP END
        q = qk
        q_dot = qk_dot
        q_ddot = qk_ddot

        yield q, q_dot, q_ddot

