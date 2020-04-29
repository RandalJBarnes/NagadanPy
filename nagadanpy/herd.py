"""
Defines and implements a contianer class of analytic elements.

Classes
-------
    Herd

Notes
-----

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    04 April 2020
"""

import numpy as np


class Herd(list):
    """
    A list of analytic elements.

    Attributes
    ----------
    None beyond what the list class provides.

    Methods
    -------
    compute_potential(x, y):
        Computes the discharge potential, Phi, at (x, y).
    compute_discharge(x, y):
        Compute the discharge vector, [Qx, Qy], at (x, y).
    compute_jacobian(x, y):
        Compute the discharge vector Jacobian at (x, y).
    """

    def __repr__(self):
        return "Herd(" + super().__repr__() + ")"

    def compute_potential(self, x, y):
        """
        Compute the discharge potential at (x,y).

        Parameters
        ----------
        x : double
            The x-coordinate of the location [m].
        y : double
            The y-coordinate of the location [m].

        Returns
        -------
        phi : double
            The discharge potential [m^3/d].
        """
        potential = 0.0
        for element in self:
            potential += element.compute_potential(x, y)
        return potential

    def compute_discharge(self, x, y):
        """
        Compute the vertically integrated discharge vector at (x, y).

        Parameters
        ----------
        x : double
            The x-coordinate of the location [m].
        y : double
            The y-coordinate of the location [m].

        Returns
        -------
        [Qx, Qy] : ndarray, dtype=double, shape=(2,).
            The vertically integrated discharge vector [m^2/d].
        """
        discharge = np.zeros((2, ), dtype=np.double)
        for element in self:
            discharge += element.compute_discharge(x, y)
        return discharge

    def compute_jacobian(self, x, y):
        """Compute the discharge vector Jacobian at (x, y).

        Parameters
        ----------
        x : double
            The x-coordinate of the location [m].
        y : double
            The y-coordinate of the location [m].

        Returns
        -------
        jac : ndarray, dtype=double, shape=(2, 2)
            The discharge vector Jacobian [m/d].
                [dQx/dx, dQy/dx]
                [dQy/dx, dQy/dy]

        Notes
        -----
        o   If (x, y) is inside the radius of the well, the discharge
            vector Jacobian is set to all zeros.
        """
        jacobian = np.zeros((2, 2), dtype=np.double)
        for element in self:
            jacobian += element.compute_jacobian(x, y)
        return jacobian
