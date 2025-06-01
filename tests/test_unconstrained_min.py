import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import numpy as np
from examples import quad1, quad2, quad3, rosenbrock, linear, triangle_exp
from unconstrained_min import optimize_gd, optimize_nt
from utils import plot_optimization_comparison

class TestUnconstrainedMinimization(unittest.TestCase):

    def check_minimizer(self, func, name, x0, max_iter=100, obj_tol=1e-6, param_tol=1e-10):
        x_min, fx, converge1, history1 = optimize_gd(func, x0, max_iter=max_iter, obj_tol=obj_tol, param_tol=param_tol)
        print(f"GD: {name} converged = {converge1}. to x = {x_min}, f(x) = {fx:.6f}")
        x_min, fx, converge2, history2 = optimize_nt(func, x0, max_iter=max_iter, obj_tol=obj_tol, param_tol=param_tol)
        print(f"NT: {name} converged = {converge2}. to x = {x_min}, f(x) = {fx:.6f}")

        plot_optimization_comparison(func, history1, history2, label1="GD", label2="NT", title=name)

        self.assertTrue(converge1 | converge2, f"{name} did not converge to a stationary point.")
        #

    def test_quad1(self):
        self.check_minimizer(quad1, "quad1", np.array([1.0, 1.0]))

    def test_quad2(self):
        self.check_minimizer(quad2, "quad2", np.array([1.0, 1.0]))

    def test_quad3(self):
        self.check_minimizer(quad3, "quad3", np.array([1.0, 1.0]))

    def test_rosenbrock(self):
        self.check_minimizer(rosenbrock, "rosenbrock", np.array([-1.0, 2.0]), max_iter=10000)

    def test_linear(self):
        self.check_minimizer(linear, "linear", np.array([1.0, 1.0]))

    def test_triangle_exp(self):
        self.check_minimizer(triangle_exp, "triangle_exp", np.array([1.0, 1.0]))

if __name__ == '__main__':
    unittest.main()
