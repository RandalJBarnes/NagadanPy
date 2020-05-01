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
    01 May 2020
"""

import numpy as np
import pytest

from nagadanpy.model import Model


@pytest.fixture
def my_model():
    base = 500.0
    conductivity = 1.0
    porosity = 0.25
    thickness = 100.0

    xo, yo = (0.0, 0.0)
    coef = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 500.0])

    wells = [
        (100.0, 200.0, 1.0, 1000.0),
        (200.0, 100.0, 1.0, 1000.0)
        ]

    mo = Model(base, conductivity, porosity, thickness, wells, xo, yo, coef)
    return mo


def test_model_compute_potential(my_model):
    x, y = [100.0, 100.0]
    Phi = my_model.compute_potential(x, y)
    assert np.isclose(Phi, 32165.8711977589, rtol=1.0e-6)


def test_model_compute_head(my_model):
    x, y = [100.0, 100.0]
    head = my_model.compute_head(x, y)
    assert np.isclose(head, 371.658711977589, rtol=1.0e-6)


def test_model_compute_discharge(my_model):
    x, y = [120.0, 160.0]
    Qx, Qy = my_model.compute_discharge(x, y)
    Qx_true, Qy_true = [-401.318309886184, -438.771830796713]
    assert np.isclose(Qx, Qx_true, rtol=1.0e-6)
    assert np.isclose(Qy, Qy_true, rtol=1.0e-6)


def test_model_compute_velocity(my_model):
    x, y = [100.0, 100.0]
    Vx, Vy = my_model.compute_velocity(x, y)
    Vx_true, Vy_true = [-11.976338022763, -11.976338022763]
    assert np.isclose(Vx, Vx_true, rtol=1.0e-6)
    assert np.isclose(Vy, Vy_true, rtol=1.0e-6)

    x, y = [120.0, 160.0]
    Vx, Vy = my_model.compute_velocity(x, y)
    Vx_true, Vy_true = [-16.052732395447, -17.550873231869]
    assert np.isclose(Vx, Vx_true, rtol=1.0e-6)
    assert np.isclose(Vy, Vy_true, rtol=1.0e-6)


def test_model_compute_fit(my_model):
    ev_true = np.array([0.9916, 0.9956, 0.9422, 171.85, 165.8, 9667.8])
    x_true, y_true = [58.52, 52.76]
    cov_true = np.array([
        [3.195e-05, -2.031e-06, -2.437e-06,  2.176e-04, -3.117e-05, -1.340e-02],
        [-2.031e-06,  1.917e-05, -3.666e-06,  1.282e-04,  1.297e-04, -1.121e-02],
        [-2.437e-06, -3.666e-06,  1.294e-05, -2.745e-05,  5.914e-05,  3.349e-03],
        [2.176e-04,  1.282e-04, -2.745e-05,  1.083e-02,  6.399e-04, -1.719e-01],
        [-3.117e-05,  1.297e-04,  5.914e-05,  6.399e-04,  7.370e-03, -6.047e-02],
        [-1.340e-02, -1.121e-02,  3.349e-03, -1.719e-01, -6.047e-02,  1.715e+01]])

    obs = list([
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

    coef_ev, coef_cov = my_model.fit_regional_flow(obs)

    assert np.isclose(my_model.xo, x_true, rtol=0.001)
    assert np.isclose(my_model.yo, y_true, rtol=0.001)
    assert np.allclose(coef_ev, ev_true, rtol=0.001)
    assert np.allclose(coef_cov, cov_true, rtol=0.001)
