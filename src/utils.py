import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_comparison(f, history1, history2, label1="Method 1", label2="Method 2",
                                  bounds=((-3, 3), (-3, 3)), levels=50, title='Optimization Comparison'):
    """
    Plot contour with optimization paths + function values over iterations.

    Parameters:
        f         : the objective function (must accept x and hessian=False)
        history1  : list of (x, f(x)) tuples for method 1
        history2  : list of (x, f(x)) tuples for method 2
        label1    : label for method 1
        label2    : label for method 2
        bounds    : ((x_min, x_max), (y_min, y_max)) range for contours
        levels    : number of contour levels
        title     : plot title
    """
    x1_vals = [x[0] for x, _ in history1]
    y1_vals = [x[1] for x, _ in history1]
    f1_vals = [fval for _, fval in history1]

    x2_vals = [x[0] for x, _ in history2]
    y2_vals = [x[1] for x, _ in history2]
    f2_vals = [fval for _, fval in history2]

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 300),
                       np.linspace(y_min, y_max, 300))

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j], _, _ = f(np.array([X[i, j], Y[i, j]]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Contour with paths
    contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(x1_vals, y1_vals, 'r-o', label=label1)
    ax1.plot(x2_vals, y2_vals, 'b-s', label=label2)
    ax1.scatter([x1_vals[0]], [y1_vals[0]], color='black', label='Start', zorder=5)
    ax1.set_title('Contour and Optimization Paths')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Function value over iterations
    ax2.plot(f1_vals, 'r-o', label=label1)
    ax2.plot(f2_vals, 'b-s', label=label2)
    ax2.set_title('Function Value vs Iteration')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
