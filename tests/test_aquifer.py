"""
Test the Aquifer class.

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


def test_constructor():
    with pytest.raises(ValueError):
        Aquifer(20, 10, -0.2, 1.0)
    with pytest.raises(ValueError):
        Aquifer(20, 10, 0.2, -1.0)


def test_setters():
    aq = Aquifer(20.0, 10.0, 0.2, 1.0)

    aq.base = 30.0
    assert aq.base == 30.0

    aq.thickness = 20.0
    assert aq.thickness == 20.0

    with pytest.raises(ValueError):
        aq.thickness = -10.0

    aq.porosity = 0.1
    assert aq.porosity == 0.1

    with pytest.raises(ValueError):
        aq.porosity = 0.0

    with pytest.raises(ValueError):
        aq.porosity = 1.0

    aq.conductivty = 1.0
    assert aq.conductivity == 1.0

    with pytest.raises(ValueError):
        aq.conductivity = -1.0


def test_elevation_to_head():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    head = aq.elevation_to_head(750.0)
    assert head == 250.0


def test_head_to_elevation():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    elevation = aq.head_to_elevation(250.0)
    assert elevation == 750.0


def test_head_to_potential():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    potential = aq.head_to_potential(250.0)
    assert potential == 20000.0


def test_potential_to_head():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    head = aq.potential_to_head(20000.0)
    assert head == 250.0


def test_elevation_to_potential():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    potential = aq.elevation_to_potential(750.0)
    assert potential == 20000.0


def test_potential_fosm():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    potential_ev, potential_std = aq.potential_fosm(750.0, 10.0)
    assert potential_ev == 20000.0
    assert potential_std == 1000.0


def test_potential_to_elevation():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    elevation = aq.potential_to_elevation(20000.0)
    assert elevation == 750.0


def test_discharge_to_velocity():
    aq = Aquifer(500.0, 100.0, 0.25, 1.0)
    head = 371.658711977589
    discharge = np.array([-299.408450569081, -299.408450569081])
    velocity = aq.discharge_to_velocity(discharge, head)
    assert math.isclose(velocity[0], -11.9763380227632, rel_tol=1.0e-6)
    assert math.isclose(velocity[1], -11.9763380227632, rel_tol=1.0e-6)
