"""
Test the RegionalFlow class.

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
    29 April 2020
"""

import math
import numpy as np
import pytest

from nagadanpy.regionalflow import RegionalFlow


def test_regionalflow_constructor():
    with pytest.raises(TypeError):
        RegionalFlow(0.0, 0.0, np.array([1.0, 1.0]))
    with pytest.raises(TypeError):
        RegionalFlow(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        RegionalFlow(0.0, 0.0, np.array([1+1j, 1.0, 1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        RegionalFlow(0.0, 0.0, np.array([np.inf, 1.0, 1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        RegionalFlow(0.0, 0.0, np.array([np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]))


def test_regionalflow_compute_potential():
    P = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r1 = RegionalFlow(100.0, 200.0, P)
    Phi = r1.compute_potential(10.0, 20.0)
    assert math.isclose(Phi, 120246.0, rel_tol=1e-6)


def test_regionalflow_compute_discharge():
    P = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r1 = RegionalFlow(100.0, 200.0, P)
    Q = r1.compute_discharge(10.0, 20.0)
    assert math.isclose(Q[0], 716.0, rel_tol=1e-6)
    assert math.isclose(Q[1], 985.0, rel_tol=1e-6)

    P = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 500.0])
    r2 = RegionalFlow(0.0, 0.0, P)
    Q = r2.compute_discharge(100.0, 100.0)
    assert math.isclose(Q[0], -301.0, rel_tol=1e-6)
    assert math.isclose(Q[1], -301.0, rel_tol=1e-6)


def test_regionalflow_compute_jacobian():
    P = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 500.0])
    re = RegionalFlow(0.0, 0.0, P)
    jacobian = re.compute_jacobian(150.0, 150.0)
    jacobian_true = np.array([[-2.0, -1.0], [-1.0, -2.0]])
    assert np.allclose(jacobian, jacobian_true, rtol=0.001)
