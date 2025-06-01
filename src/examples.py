import numpy as np

def quad1(x, hessian=False):
    Q = np.eye(2)
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian else None
    return f, g, h

def quad2(x, hessian=False):
    Q = np.diag([1, 100])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian else None
    return f, g, h

def quad3(x, hessian=False):
    theta = np.pi / 6  # 30 degrees
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Q_base = np.diag([100, 1])
    Q = R.T @ Q_base @ R
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian else None
    return f, g, h

def rosenbrock(x, hessian=False):
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    g = np.array([
        -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1),
        200 * (x2 - x1 ** 2)
    ])
    h = None
    if hessian:
        h = np.array([
            [1200 * x1 ** 2 - 400 * x2 + 2, -400 * x1],
            [-400 * x1, 200]
        ])
    return f, g, h

def linear(x, hessian=False):
    a = np.array([1.0, 2.0])  # any non-zero vector
    f = a.T @ x
    g = a
    h = np.zeros((2, 2)) if hessian else None
    return f, g, h

def triangle_exp(x, hessian=False):
    x1, x2 = x[0], x[1]
    e1 = np.exp(x1 + 3 * x2 - 0.1)
    e2 = np.exp(x1 - 3 * x2 - 0.1)
    e3 = np.exp(-x1 - 0.1)
    f = e1 + e2 + e3

    g = np.array([
        e1 + e2 - e3,
        3 * e1 - 3 * e2
    ])

    h = None
    if hessian:
        h = np.array([
            [e1 + e2 + e3, 3 * e1 - 3 * e2],
            [3 * e1 - 3 * e2, 9 * e1 + 9 * e2]
        ])

    return f, g, h
