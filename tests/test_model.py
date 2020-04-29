"""
Test the Model class.

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

from aquifer import Aquifer
from herd import Herd
from model import Model
from regionalflow import RegionalFlow
from well import Well


@pytest.fixture
def my_model():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)

    we1 = Well(100.0, 200.0, 1.0, 1000.0)
    we2 = Well(200.0, 100.0, 1.0, 1000.0)
    wells = Herd([we1, we2])

    P = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 500.0])
    re = RegionalFlow(0.0, 0.0, P)

    return Model(aq, re, wells)


def test_model_compute_potential(my_model):
    Phi = my_model.compute_potential(100.0, 100.0)
    assert math.isclose(Phi, 32165.8711977589, rel_tol=1.0e-6)


def test_model_compute_head(my_model):
    head = my_model.compute_head(100.0, 100.0)
    assert math.isclose(head, 371.658711977589, rel_tol=1.0e-6)


def test_model_compute_elevation(my_model):
    elevation = my_model.compute_elevation(100.0, 100.0)
    assert math.isclose(elevation, 371.658711977589+500.0, rel_tol=1.0e-6)


def test_model_compute_discharge(my_model):
    discharge = my_model.compute_discharge(120.0, 160.0)
    discharge_true = np.array([-401.318309886184, -438.771830796713])
    assert np.allclose(discharge, discharge_true, rtol=1.0e-6)


def test_compute_jacobian(my_model):
    jacobian = my_model.compute_jacobian(110.0, 195.0)
    jacobian_true = np.array([[-1.2365, -2.0277], [-2.0279, -2.7634]])
    assert np.allclose(jacobian, jacobian_true, rtol=0.001)


def test_compute_velocity(my_model):
    velocity = my_model.compute_velocity(100.0, 100.0)
    velocity_true = np.array([[-11.976338022763, -11.976338022763]])
    assert np.allclose(velocity, velocity_true, rtol=1.0e-6)

    velocity = my_model.compute_velocity(120.0, 160.0)
    velocity_true = np.array([[-16.052732395447, -17.550873231869]])
    assert np.allclose(velocity, velocity_true, rtol=1.0e-6)


def test_model_compute_fit():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    re = RegionalFlow(0.0, 0.0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    we1 = Well(0.0, 0.0, 1.0, 1000.0)
    we2 = Well(100.0, 100.0, 1.0, 1000.0)
    he = Herd([we1, we2])
    mo = Model(aq, re, he)

    P_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 500.0])
    obs = np.array([
        [23.00, 11.00, 573.64, 0.10],
        [24.00, 85.00, 668.55, 0.10],
        [26.00, 80.00, 661.58, 0.10],
        [28.00, 65.00, 637.97, 0.10],
        [37.00, 50.00, 626.62, 0.10],
        [41.00, 21.00, 598.85, 0.10],
        [42.00, 53.00, 637.51, 0.10],
        [42.00, 74.00, 673.32, 0.10],
        [45.00, 70.00, 670.52, 0.10],
        [46.00, 15.00, 599.43, 0.10],
        [52.00, 76.00, 694.14, 0.10],
        [58.00, 90.00, 736.75, 0.10],
        [64.00, 22.00, 629.54, 0.10],
        [71.00, 19.00, 637.34, 0.10],
        [72.00, 36.00, 660.54, 0.10],
        [72.00, 55.00, 691.45, 0.10],
        [74.00, 50.00, 686.57, 0.10],
        [75.00, 18.00, 642.92, 0.10],
        [76.00, 43.00, 678.80, 0.10],
        [77.00, 79.00, 752.05, 0.10],
        [79.00, 66.00, 727.81, 0.10],
        [81.00, 81.00, 766.23, 0.10],
        [82.00, 77.00, 759.15, 0.10],
        [86.00, 26.00, 673.24, 0.10],
        [90.00, 57.00, 734.72, 0.10]])

    P_ev, P_cov = mo.fit_coefficients(0.0, 0.0, obs)
    assert np.allclose(P_ev, P_true, rtol=0.01)
