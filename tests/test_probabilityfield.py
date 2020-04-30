"""
Test the ProbabilityField class.

Notes
-----

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    30 April 2020
"""

import numpy as np
import pytest

from nagadanpy.probabilityfield import ProbabilityField, RangeError


def test_probabilityfield_constructor():
    with pytest.raises(RangeError):
        ProbabilityField(0, 0)

    pf = ProbabilityField(1, 1)
    assert np.isclose(pf.deltax, 1, rtol=1e-9)
    assert np.isclose(pf.deltay, 1, rtol=1e-9)


def test_probabilityfield_expand():
    pf = ProbabilityField(1, 1)

    with pytest.raises(RangeError):
        pf.expand(1, 0, 1, 0)

    pf.expand(100.0, 200.0, 50.0, 100.0)
    assert pf.nrows == 53
    assert pf.ncols == 103
    assert np.isclose(pf.xmin, 99.0, rtol=1e-9)
    assert np.isclose(pf.xmax, 201.0, rtol=1e-9)
    assert np.isclose(pf.ymin, 49.0, rtol=1e-9)
    assert np.isclose(pf.ymax, 101.0, rtol=1e-9)

    pf.expand(150.0, 175.0, 75.0, 100.0)
    assert pf.nrows == 53
    assert pf.ncols == 103


def test_probabilityfield_distancesquared():
    assert np.isclose(ProbabilityField.distancesquared(0, 1, 1, 0, 0, 0), 0.5, atol=1e-9)

# TODO: add tests for insert and rasterize.
