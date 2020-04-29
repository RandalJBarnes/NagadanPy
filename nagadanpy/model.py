"""
Defines and implements an Oneka-type analytic element model.

Classes
-------
    Model

Notes
-----
o   An Oneka-type analytic element model includes three analytic element components:
    uniform flow, uniform recharge, and a set of discharge-specified wells.

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

from aquifer import Aquifer
from herd import Herd
from regionalflow import RegionalFlow
from well import Well


class Model:
    """
    An Oneka-type analytic element model.

    Attributes
    ----------
    aquifer : Aquifer
        an instance of the Aquifer class.
    regionalflow : RegionalFlow
        an instance of the RegionalFlow class.
    wells : Herd
        list of instances of the Well class.

    Methods
    -------
    compute_potential(x, y):
        Computes the discharge potential, Phi, at (x, y).
    compute_discharge(x, y):
        Compute the discharge vector, [Qx, Qy], at (x, y).
    compute_jacobian(x, y):
        Compute the discharge vector Jacobian at (x, y).
    compute_head(x, y):
        Compute the piezometric head at (x, y).
    compute_elevation(x, y):
        Compute the elevation of the static water level at (x, y).
    compute_velocity(x, y):
        Compute the seepage velocity vector, [Vx, Vy], at (x, y).
    fit_coefficients(xo, yo, obs):
        Set the model's regional flow coefficients using the local
        origin, <xo, yo>, the observed heads, <obs>, and a weighted
        least squares fit.
    """

    def __repr__(self):
        return "Model({0.aquifer!r}, {0.regionalflow!r}, {0.wells!r})".format(self)

    def __str__(self):
        return "Model(\n\t{0.aquifer}, \n\t{0.regionalflow}, \n\t{0.wells} \n)".format(self)

    def __init__(self, aquifer, regionalflow, wells):
        """
        Initializes all of the attributes for a Model object.

        Parameters
        ----------
        aquifer : Aquifer
            an instance of the Aquifer class.
        regionalflow : RegionalFlow
            an instance of the RegionalFlow class.
        wells : herd
            herd of instances of the Well class.
        """
        self.aquifer = aquifer
        self.regionalflow = regionalflow
        self.wells = wells

    @property
    def aquifer(self):
        return self._aquifer

    @aquifer.setter
    def aquifer(self, aquifer):
        if not isinstance(aquifer, Aquifer):
            raise TypeError("aquifer is of type Aquifer.")
        self._aquifer = aquifer

    @property
    def regionalflow(self):
        return self._regionalflow

    @regionalflow.setter
    def regionalflow(self, regionalflow):
        if not isinstance(regionalflow, RegionalFlow):
            raise TypeError("regionalflowis of type RegionalFlow.")
        self._regionalflow = regionalflow

    @property
    def wells(self):
        return self._wells

    @wells.setter
    def wells(self, wells):
        if not isinstance(wells, Herd) or not all(isinstance(w, Well) for w in wells):
            raise TypeError("wells is a Herd of type Well.")
        self._wells = wells

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
        Phi : double
            The discharge potential [m^3/d].
        """
        return self.regionalflow.compute_potential(x, y) + self.wells.compute_potential(x, y)

    def compute_discharge(self, x, y):
        """
        Compute the vertically integrated discharge vector at (x, y) [m^2/d].

        Parameters
        ----------
        x : double
            The x-coordinate of the location [m].
        y : double
            The y-coordinate of the location [m].

        Returns
        -------
        [Qx, Qy] : ndarray, shape=(2,).
            The vertically integrated discharge vector [m^2/d].
        """
        return self.regionalflow.compute_discharge(x, y) + self.wells.compute_discharge(x, y)

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
        jac : ndarray, dtype=double, shape = (2, 2)
            The discharge vector Jacobian [m/d].
                [dQx/dx, dQy/dx]
                [dQy/dx, dQy/dy]
        """
        return self.regionalflow.compute_jacobian(x, y) + self.wells.compute_jacobian(x, y)

    def compute_head(self, x, y):
        """
        Compute the piezometric head at (x, y).

        Parameters
        ----------
        x : double
            The x-coordinate of the location [m].
        y : double
            The y-coordinate of the location [m].

        Returns
        -------
        head : double
            The piezometric head measured from the base of the aquifer [m].
        """
        potential = self.compute_potential(x, y)
        head = self.aquifer.potential_to_head(potential)
        return head

    def compute_elevation(self, x, y):
        """
        Compute the elevation of the static water level at (x, y).

        Parameters
        ----------
        x : double
            The x-coordinate of the location [m].
        y : double
            The y-coordinate of the location [m].

        Returns
        -------
        elevation : double
            The elevation of the static water level [m].
        """
        potential = self.compute_potential(x, y)
        elevation = self.aquifer.potential_to_elevation(potential)
        return elevation

    def compute_velocity(self, x, y):
        """
        Compute the vertically average seepage velocity vector at (x, y).

        Parameters
        ----------
        x : double
            The x-coordinate of the location [m].
        y : double
            The y-coordinate of the location [m].

        Returns
        -------
        [Vx, Vy] : ndarray, dtype=double, shape=(2,).
            The vertically averaged seepage velocity vector [m^2/d].
        """
        discharge = self.compute_discharge(x, y)
        head = self.compute_head(x, y)
        velocity = self.aquifer.discharge_to_velocity(discharge, head)
        return velocity

    def fit_coefficients(self, xo, yo, obs):
        """
        Set the model's regional flow coefficients using the local
        origin, <xo, yo>, the observed heads, <obs>, and a weighted
        least squares fit.

        Parameters
        ----------
        xo : double
            The x-coordinate of the local origin [m].
        yo : double
            The y-coordinate of the local origin [m].
        obs : ndarray, shape(:, 4)
            The array of observations. There is one row for each observation, and
            each row contains four values: [x, y, z_ev, z_std].
                x : double
                    The x-coordinate of the observation [m].
                y : double
                    The y-coordinate of the observation [m].
                z_ev : double
                    The expected value of the observed static water level elevation [m].
                z_std : double
                    The standard deviation of the observed static water level elevation [m].

        Returns
        -------
        An ordered pair of ndarray (coef_ev, coef_cov).
            coef_ev : ndarray, dtype=double, shape=(6, ).
                The expected value vector for the model's fitted regional flow coefficients.
            coef_cov : ndarray, dtype=double, shape=(6, 6).
                The variance/covariance matrix for the model's fitted regional flow coefficients.

        Notes
        -----
        o The caller should eliminate observations that are too close to pumping wells.
        """

        self.regionalflow.x = xo
        self.regionalflow.y = yo
        self.regionalflow.coef = np.zeros(6,)

        nobs = obs.shape[0]
        A = np.zeros([nobs, 6])
        b = np.zeros([nobs, 1])
        W = np.zeros([nobs, nobs])

        for i in range(nobs):
            x, y, z_ev, z_std = obs[i, :]

            pot_ev, pot_std = self.aquifer.potential_fosm(z_ev, z_std)
            W[i, i] = 1/pot_std

            A[i, :] = [(x-xo)**2, (y-yo)**2, (x-xo)*(y-yo), (x-xo), (y-yo), 1]

            pot_wells = self.compute_potential(x, y)
            b[i] = pot_ev - pot_wells

        WA = np.matmul(W, A)
        Wb = np.matmul(W, b)
        coef_ev = np.linalg.lstsq(WA, Wb, rcond=-1)[0]
        coef_ev = np.reshape(coef_ev, [6, ])

        AWWA = np.matmul(WA.T, WA)
        coef_cov = np.linalg.inv(AWWA)

        self.regionalflow.coef = coef_ev
        return(coef_ev, coef_cov)
