import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import solve


def newmarkbeta_solver(m, c, k, f_func, init_state, t_stop, dt, beta=0.25, gamma=0.5):
    """Newmark-Beta implicit solver for linear/weakly-nonlinear dynamics.
    EOM: M q̈ + C q̇ + K q = f(q, q̇, t)
    Yields (q, q_dot, q_ddot) at each time step.
    """
    q, q_dot = init_state
    q_ddot = np.zeros_like(q)

    for t in np.arange(0, t_stop, dt):
        qp, qp_dot, qp_ddot = q, q_dot, q_ddot

        a = 1 / (beta * dt ** 2) * m + gamma / (beta * dt) * c + k
        b = (m @ ((1 / (2 * beta) - 1) * qp_ddot
                  + (1 / (beta * dt)) * qp_dot
                  + (1 / (beta * dt ** 2)) * qp)
             + c @ ((gamma / (2 * beta) - 1) * dt * qp_ddot
                    + (gamma / beta - 1) * qp_dot
                    + (gamma / (beta * dt)) * qp))

        # Predictor
        pred_ddot = q_ddot
        q = qp + dt * qp_dot + (dt ** 2) / 2 * ((1 - 2 * beta) * qp_ddot + 2 * beta * pred_ddot)
        q_dot = qp_dot + dt * ((1 - gamma) * qp_ddot + gamma * pred_ddot)

        # Corrector loop
        conv = False
        while not conv:
            qe = q
            f = f_func(qe, q_dot, t + dt)
            q = solve(a, b + f)
            q_dot = (gamma / (beta * dt) * (q - qp)
                     + (1 - gamma / beta) * qp_dot
                     + (1 - gamma / (2 * beta)) * dt * qp_ddot)
            conv = np.linalg.norm(q - qe) < 1e-6

        q_ddot = (1 / (beta * dt ** 2) * (q - qp - dt * qp_dot)
                  + (beta - 0.5) / beta * qp_ddot)

        yield q, q_dot, q_ddot


def _nnr_step(m, c, f_int_func, f_ext_func, j_func, state, t_new, dt,
              beta=0.25, gamma=0.5, conv_err=1e-4, max_iter=100):
    """
    Single Newmark/Newton-Raphson step from the converged state (q, q_dot, q_ddot)
    at time t to t_new = t + dt.

    EOM: M q̈ + C q̇ + f_int(q) = f_ext(q, q̇, t)
    Residual: R = f_ext - f_int - M q̈ - C q̇ = 0

    The Newmark predictor stage rearranges the standard Newmark formulas to isolate out
    the unknown new-step acceleration. Based on the previous converged state, the predictors for t_new are:
        q_pred = q(t) + Δt·q̇(t) + Δt²(½ - β)·q̈(t)
        q̇_pred = q̇(t) + Δt(1 - γ)·q̈(t)
    These are the standard Newmark expressions for the next time step, with the
    unknown new acceleration term factored out as a correction to be determined by
    Newton iteration. The predictor-relative acceleration correction is initialized to zero,
    and the iteration will solve for the consistent acceleration that satisfies the EOM at t_new.

    The Newton stage then iterates on a displacement correction Δq. Newmark
    kinematics tie q̈ and q̇ to q, at any given k-th iteration for the new state, through:
        q̈_k = m_coef * (q_k - q_pred)
        q̇_k = q̇_pred + c_coef * (q_k - q_pred)
    where m_coef = 1/(β·Δt²) and c_coef = γ/(β·Δt).
    The residual R depends on the trial state (q_trial, q̇_trial, q̈_trial), and Newton solves K̂·Δq = R,
    where K̂ = M·m_coef + C·c_coef + ∂f_int/∂q is the tangent stiffness matrix.
    Each Newton step updates:
        q_trial += Δq
        q̇_trial += c_coef * Δq
        q̈_trial += m_coef * Δq
    This ensures consistency between displacement, velocity, and acceleration
    throughout the iteration. Convergence occurs when |Δq| < conv_err.
    """
    q, q_dot, q_ddot = state

    m_coef = 1.0 / (beta * dt ** 2)   # dq̈/dq from Newmark
    c_coef = gamma / (beta * dt)       # dq̇/dq from Newmark

    # Standard Newmark predictors for position and velocity.
    # They provide the initial guess for the Newton-Raphson iteration at t + dt.
    # The trial displacement is then corrected by Δq, with q̇ and q̈ updated
    # consistently from the same increment.
    q_trial = q + dt * q_dot + dt ** 2 * (0.5 - beta) * q_ddot
    q_dot_trial = q_dot + dt * (1.0 - gamma) * q_ddot
    q_ddot_trial = np.zeros_like(q_trial)  # predictor-relative acceleration correction starts at zero

    delta_q = np.nan

    for iteration in range(max_iter):
        f_ext = f_ext_func(q_trial, q_dot_trial, t_new)
        f_int = f_int_func(q_trial)

        residual = f_ext - f_int - m @ q_ddot_trial - c @ q_dot_trial

        k_tangent = np.asarray(j_func(jnp.array(q_trial, dtype=float)))
        k_hat = m_coef * m + c_coef * c + k_tangent

        delta_q = solve(k_hat, np.asarray(residual))

        # Consistent Newmark update of all three kinematic quantities
        q_trial = q_trial + delta_q
        q_dot_trial = q_dot_trial + c_coef * delta_q
        q_ddot_trial = q_ddot_trial + m_coef * delta_q

        #print(np.linalg.norm(delta_q))
        if np.linalg.norm(delta_q) < conv_err:
            break
    else:
        raise RuntimeError(
            f"NNR did not converge in {max_iter} iterations at t={t_new:.4f}; "
            f"|Δq|={np.linalg.norm(delta_q):.3e}"
        )

    return q_trial, q_dot_trial, q_ddot_trial


def nnr_solver(m, c, f_int_func, f_ext_func, init_state, t_stop, dt,
               beta=0.25, gamma=0.5, conv_err=1e-4):
    """
    Newmark/Newton-Raphson generator solver for nonlinear structural dynamics.

    EOM: M q̈ + C q̇ + f_int(q) = f_ext(q, q̇, t)

    Yields (q, q_dot, q_ddot) at t = dt, 2·dt, ..., t_stop.
    The time-stepping loop lives here; _nnr_step handles a single step.
    """
    q, q_dot = init_state
    # Bootstrap initial acceleration from EOM at t=0
    q_ddot = np.linalg.solve(
        m, np.asarray(f_ext_func(q, q_dot, 0.0))
           - np.asarray(f_int_func(q))
           - c @ q_dot
    )

    # Build jacobian function once; JAX JIT-compiles on first call and caches thereafter
    j_func = jax.jacobian(f_int_func)

    state = (q, q_dot, q_ddot)

    for t in np.arange(0, t_stop, dt):
        state = _nnr_step(
            m, c, f_int_func, f_ext_func, j_func,
            state, t_new=t + dt, dt=dt,
            beta=beta, gamma=gamma, conv_err=conv_err,
        )
        yield state
