"""
Defines and implements an Oneka-type regional flow analytic element.

Classes
-------
    RegionalFlow

Notes
-----
o   An Oneka-type regional flow element implicitly includes uniform flow and uniform recharge.
    The discharge potential is given by the quadratic form:

        Phi = Ax^2 + By^2 + Cxy + Dx + Ey + F

    Physically, this is a quadrtic mound with elliptical contours.

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


class RegionalFlow():
    """
    An Oneka-type regional flow analytic element.

    Attributes
    ----------
    x : double
        The x-coordinate of the well center [m].
    y : double
        The y-coordinate of the well center [m].
    coef : ndarray, dtype=double, shape=(6, )
        The six defining coefficients, A through F.

    Methods
    -------
    compute_potential(x, y):
        Computes the discharge potential, Phi, at (x, y).
    compute_discharge(x, y):
        Compute the discharge vector, [Qx, Qy], at (x, y).
    compute_jacobian(x, y):
        Compute the discharge vector Jacobian at (x, y).
    """

    def __init__(self, x, y, coef):
        """
        Initializes all of the attributes for a RegionalFlow object.

        Parameters
        ----------
        x : double
            The x-coordinate of the well center [m].
        y : double
            The y-coordinate of the well center [m].
        coef : ndarray, dtype=double, shape=(6, )
            The six defining coefficients, A through F.
        """

        self.x = x
        self.y = y
        self.coef = coef

    def __repr__(self):
        return "RegionalFlow({0.x}, {0.y}, {0.coef!r})".format(self)

    def __str__(self):
        return "RegionalFlow({0.x}, {0.y}, {0.coef})".format(self)

    @property
    def coef(self):
        return self._coef

    @coef.setter
    def coef(self, coef):
        if not isinstance(coef, np.ndarray) or coef.shape != (6,):
            raise TypeError("{coef} must be an numpy.ndarray of shape (6,).")
        if not np.all(np.isfinite(coef)) or not np.all(np.isreal(coef)):
            raise ValueError("All {coef} must be real and finite.")
        self._coef = coef

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

        Notes
        -----
        o   If (x,y) is inside the radius of the well, the discharge potential is set to the
            discharge potential at the radius.
        """

        dx = x - self.x
        dy = y - self.y

        phi = (self.coef[0]*dx**2 + self.coef[1]*dy**2 + self.coef[2]*dx*dy
               + self.coef[3]*dx + self.coef[4]*dy + self.coef[5])
        return phi

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
        [Qx, Qy] : ndarray, dtype=double, shape=(2,)
            The vertically integrated discharge vector [m^2/d].
        """

        dx = x - self.x
        dy = y - self.y

        Qx = -(2.0*self.coef[0]*dx + self.coef[2]*dy + self.coef[3])
        Qy = -(2.0*self.coef[1]*dy + self.coef[2]*dx + self.coef[4])
        return np.array([Qx, Qy], dtype=np.double)

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
        """

        dQxdx = -2.0*self.coef[0]
        dQxdy = -self.coef[2]
        dQydx = -self.coef[2]
        dQydy = -2.0*self.coef[1]
        jacobian = np.array([[dQxdx, dQxdy], [dQydx, dQydy]], dtype=np.double)
        return jacobian
