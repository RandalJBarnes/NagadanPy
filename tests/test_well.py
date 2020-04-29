"""
Test the Well class.

Notes
-----
o   The specific test values were computed using the MatLab code
    from the "Object Based Analytic Elements" project.

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    04 April 2020
"""

import math
import numpy as np
import pytest

from well import Well


def test_constructor():
    with pytest.raises(ValueError):
        Well(100.0, 200.0, -1.0, 10.0)


def test_compute_potential():
    w1 = Well(100.0, 200.0, 1.0, 10.0)
    Phi = w1.compute_potential(10.0, 20.0)
    assert math.isclose(Phi, 8.44241951687468, rel_tol=1e-6)


def test_compute_discharge():
    w1 = Well(100.0, 200.0, 1.0, 1000.0)
    discharge = w1.compute_discharge(110.0, 195.0)
    discharge_true = np.array([[-12.732395447352, 6.366197723676]])
    assert np.allclose(discharge, discharge_true, rtol=1.0e-6)


def test_compute_jacobian():
    we = Well(100.0, 200.0, 1.0, 1000.0)
    jacobian = we.compute_jacobian(110.0, 195.0)
    jacobian_true = np.array([[0.7639, -1.0184], [-1.01868, -0.7639]])
    assert np.allclose(jacobian, jacobian_true, rtol=0.001)
