"""
Test the Herd class.

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

from herd import Herd
from well import Well


@pytest.fixture
def my_herd():
    w1 = Well(100.0, 200.0, 1.0, 1000.0)
    w2 = Well(200.0, 100.0, 1.0, 1000.0)
    he = Herd([w1, w2])

    w3 = Well(200.0, 200.0, 1.0, 1000.0)
    he.append(w3)

    return he


def test_compute_potential(my_herd):
    phi = my_herd.compute_potential(150.0, 150.0)
    assert math.isclose(phi, 2033.33009652379, rel_tol=1e-6)


def test_compute_discharge(my_herd):
    discharge = my_herd.compute_discharge(150.0, 150.0)
    discharge_true = np.array([1.59154943091895, 1.59154943091895])
    assert np.allclose(discharge, discharge_true, rtol=1.0e-6)


def test_compute_jacobian(my_herd):
    jacobian = my_herd.compute_jacobian(110.0, 195.0)
    jacobian_true = np.array([[ 0.78290949, -1.02570201], [-1.02570201, -0.78290949]])
    assert np.allclose(jacobian, jacobian_true, rtol=0.001)
