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
    05 May 2020
"""

import math
import numpy as np

import nagadanpy.boomerang
from nagadanpy.boomerang import compute_kldiv

"""
def test_boomerang():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    re = RegionalFlow(0.0, 0.0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    we1 = Well(0.0, 0.0, 1.0, 1000.0)
    we2 = Well(100.0, 100.0, 1.0, 1000.0)
    he = Herd([we1, we2])
#    mo = Model(aq, re, he)

    obs = [
        (23.00, 11.00, 573.64, 0.10),
        (24.00, 85.00, 668.55, 0.10),
        (26.00, 80.00, 661.58, 0.10),
        (28.00, 65.00, 637.97, 0.10),
        (37.00, 50.00, 626.62, 0.10),
        (41.00, 21.00, 598.85, 0.10),
        (42.00, 53.00, 637.51, 0.10),
        (42.00, 74.00, 673.32, 0.10),
        (45.00, 70.00, 670.52, 0.10),
        (46.00, 15.00, 599.43, 0.10),
        (52.00, 76.00, 694.14, 0.10),
        (58.00, 90.00, 736.75, 0.10),
        (64.00, 22.00, 629.54, 0.10),
        (71.00, 19.00, 637.34, 0.10),
        (72.00, 36.00, 660.54, 0.10),
        (72.00, 55.00, 691.45, 0.10),
        (74.00, 50.00, 686.57, 0.10),
        (75.00, 18.00, 642.92, 0.10),
        (76.00, 43.00, 678.80, 0.10),
        (77.00, 79.00, 752.05, 0.10),
        (79.00, 66.00, 727.81, 0.10),
        (81.00, 81.00, 766.23, 0.10),
        (82.00, 77.00, 759.15, 0.10),
        (86.00, 26.00, 673.24, 0.10),
        (90.00, 57.00, 734.72, 0.10)]

    kldiv_one, kldiv_two = boomerang(mo, obs)
    # TODO: add a answer to compare with.
"""


def test_boomerang_compute_kldiv():
    muf = np.array([[109], [111], [-86], [8], [-121], [-111]])
    covf = np.array([
        [92, -6, 11, -10, 13, 2],
        [-6, 60, 12, -23, 6, 6],
        [11, 12, 54, -3, 8, 12],
        [-10, -23, -3, 87, -9, -15],
        [13, 6, 8, -9, 61, -3],
        [2, 6, 12, -15, -3, 58]
        ])
    covf_inv = np.linalg.inv(covf)

    mug = np.array([[-1], [153], [-77], [37], [-23], [112]])
    covg = np.array([
        [113, -23, -18, 35, 24, -3],
        [-23, 144, 33, 3, -22, 7],
        [-18, 33, 181, -14, 27, -31],
        [35, 3, -14, 59, 1, -6],
        [24, -22, 27, 1, 194, -16],
        [-3, 7, -31, -6, -16, 87]
        ])

    kldiv = compute_kldiv(muf, covf, covf_inv, mug, covg)
    assert math.isclose(kldiv, 1039.01850317409, rel_tol=1e-6)
