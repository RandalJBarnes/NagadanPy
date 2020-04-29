"""
Defines and implements a discharge specified well analytic element.

Classes
-------
    Well

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


class Well():
    """
    A discharge-specified well analytic element.

    Attributes
    ----------
    x : double
        The x-coordinate of the well center [m].
    y : double
        The y-coordinate of the well center [m].
    radius : double
        The radius of the well [m]. radius > 0.
    discharge : double
        The discharge of the well [m^3/d].

    Methods
    -------
    compute_potential(x, y):
        Computes the discharge potential, Phi, at (x, y).
    compute_discharge(x, y):
        Compute the discharge vector, [Qx, Qy], at (x, y).
    compute_jacobian(x, y):
        Compute the discharge vector Jacobian at (x, y).
    """

    def __init__(self, x, y, radius, discharge):
        """
        Initializes all of the attributes for a Well object.

        Parameters
        ----------
        x : double
            The x-coordinate of the well center [m].
        y : double
            The y-coordinate of the well center [m].
        radius : double
            The radius of the well [m]. radius > 0.
        discharge : double
            The discharge of the well [m^3/d].
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.discharge = discharge

    def __repr__(self):
        return "Well({0.x}, {0.y}, {0.radius}, {0.discharge})".format(self)

    def __str__(self):
        return self.__repr()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        if radius <= 0.0:
            raise ValueError("radius > 0")
        self._radius = radius

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
        o   If (x,y) is inside the radius of the well, the discharge
            potential is set to the discharge potential at the radius.
        """
        r = np.hypot(x - self.x, y - self.y)
        if r < self.radius:
            phi = self.discharge/(2.0*np.pi) * np.log(self.radius)
        else:
            phi = self.discharge/(2.0*np.pi) * np.log(r)
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
        [Qx, Qy] : ndarray, dtype=double, shape=(2,).
            The vertically integrated discharge vector [m^2/d].

        Notes
        -----
        o   If (x, y) is inside the radius of the well, the verticlly
            integrated discharge vector is set to all zeros.
        """
        dx = x - self.x
        dy = y - self.y

        r = np.hypot(dx, dy)
        if r < self.radius:
            Qx = 0.0
            Qy = 0.0
        else:
            Qx = -self.discharge/(2.0*np.pi) * dx/r**2
            Qy = -self.discharge/(2.0*np.pi) * dy/r**2
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

        Notes
        -----
        o   If (x, y) is inside the radius of the well, the discharge
            vector Jacobian is set to all zeros.
        """
        dx = x - self.x
        dy = y - self.y

        r = np.hypot(dx, dy)
        if r < self.radius:
            jacobian = np.zeros((2, 2), dtype=np.double)
        else:
            a = dx*dx - dy*dy
            b = 2*dx*dy
            c = self.discharge/(2.0*np.pi) / r**4

            jacobian = np.array([[a*c, b*c], [b*c, -a*c]], dtype=np.double)
        return jacobian
