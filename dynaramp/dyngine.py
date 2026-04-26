import numpy as np
from scipy.linalg import solve

def newmarkbeta_integrator(m_func, c_func, k_func, f_func, init_state, dt, beta=0.25, gamma=0.5):
    # Initializing problem
    q = init_state  # generalized coordinates
    m = m_func(q)  # mass matrix
    k = k_func(q)  # stiffness matrix
    c = c_func(q)  # damping matrix

    # Solve Ax = B problem
    a = 1/(beta * dt**2) * m + gamma/(beta * dt) * c + k
    b = f_func
    solve(a, b)
