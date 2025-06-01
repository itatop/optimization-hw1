
import numpy as np


def optimize_gd(f, x0, obj_tol=1e-6, param_tol=1e-10, max_iter=100):

    alpha_init=1.0
    rho=0.5
    c=0.01

    x = x0.copy()
    history = []

    for i in range(max_iter):
        fx, grad, _ = f(x, False)
        history.append((x.copy(), fx))

        # Compute descent direction
        d = -grad

        # Backtracking line search
        alpha = alpha_init
        while True:
            x_new = x + alpha * d
            fx_new, _, _ = f(x_new)
            if fx_new <= fx + c * alpha * np.dot(grad, d):
                break
            alpha *= rho  # reduce step size

        # Check for convergence
        if np.linalg.norm(x_new - x) < param_tol:
            print(f"Converged: parameter change < {param_tol}")
            return x_new, fx_new, True, history
        if abs(fx_new - fx) < obj_tol:
            print(f"Converged: objective change < {obj_tol}")
            return x_new, fx_new, True, history

        x = x_new

    # Failed to converge
    fx, _, _ = f(x, False)
    return x, fx, False, history

def optimize_nt(f, x0, obj_tol=1e-6, param_tol=1e-10, max_iter=100):

    alpha_init=1.0
    rho=0.5
    c=0.01

    x = x0.copy()
    history = []

    for i in range(max_iter):
        fx, grad, h = f(x, True)
        history.append((x.copy(), fx))

        # Compute descent direction
        d = -grad

        p = np.linalg.solve(h, d)

        # Backtracking line search
        alpha = alpha_init
        while True:
            x_new = x + alpha * p
            fx_new, _, _ = f(x_new)
            if fx_new <= fx + c * alpha * np.dot(grad, p):
                break
            alpha *= rho  # reduce step size

        # Check for convergence
        if np.linalg.norm(x_new - x) < param_tol:
            print(f"Converged: parameter change < {param_tol}")
            return x_new, fx_new, True, history
        if abs(fx_new - fx) < obj_tol:
            print(f"Converged: objective change < {obj_tol}")
            return x_new, fx_new, True, history

        x = x_new

    # Failed to converge
    fx, _, _ = f(x, False)
    return x, fx, False, history
