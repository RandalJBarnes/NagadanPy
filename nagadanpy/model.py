"""
Defines and implements an Oneka-type analytic element model.

Classes
-------
    Model

Exceptions
----------
    Error
    RangeError
    AquiferError

Notes
-----
o   An Oneka-type analytic element model includes three analytic element
    components: uniform flow, uniform recharge, and a set of discharge-
    specified wells.

o   How to line-by-line profile:
    - Add the decorator "@profile" to the functions of interest.
    - kernprof -l -v <driver>.py
    - python -m line_profiler <driver>.py.lprof > results.txt

Authors
-------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

    Richard Soule
    Source Water Protection
    Minnesota Department of Health

Version
-------
    02 Model 2020
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


class Error(Exception):
    """Base class for module errors."""
    pass


class RangeError(Error):
    """Passed argument out of valid range."""
    pass


class AquiferError(Error):
    """The aquifer is dry: head <= 0."""
    pass


# ---------------------------------------------------------
class Model:
    """
    An Oneka-type analytic element model.

    Attributes
    ----------
    base : float
        The base elevation of the aquifer [m].

    conductivity : float
        The hydraulic conductivity of the aquifer [m/d]. conductivity > 0.

    porosity : float
        The porosity of the aquifer []. 0 < porosity < 1.

    thickness : float
        The thickness of the aquifer [m]. thickness > 0.

    wells : list
        The list of well tuples. Each well tuple has four components.
            xw : float
                The x-coordinate of the well [m].
            yw : float
                The y-coordinate of the well [m].
            rw : float
                The radius of the well [m]. rw > 0.
            qw : float
                The discharge of the well [m^3/d].

    xo : float, optional
        The x-coordinate of the local origin [m] for the regional flow
        component of the Oneka model.

    yo : float, optional
        The y-coordinate of the local origin [m] for the regional flow
        component of the Oneka model.

    coef : ndarray, shape=(6, ), dtype=float, optional.
        The six coefficient for the regional flow: A, B, C, D, E, F.

    Methods
    -------
    compute_potential(x, y):
        Computes the discharge potential [m^3/d] at (x, y).

    compute_discharge(x, y):
        Compute the vertically integrated discharge [m^2/d] at (x, y).

    compute_head(xy):
        Compute the piezometric head [m]  at (x, y).

    compute_velocity(x, y):
        Compute the vertically averaged seepage velocity vector [m/d] at (x, y).

    compute_potential_fosm(self, elevation_ev, elevation_std):
        Compute the avg and std of the discharge potential from the avg and std
        of the elevation, using a first-order-second-moment approximation.

    fit_coefficients(self, obs):
        Set the regionalflow's origin to the cetroid of the obs, then
        fit the regionalflow's coefficient using weighted least squares.
    """

    # ---------------------------------
    def __init__(self, base, conductivity, porosity, thickness, wells,
                 xo=0, yo=0, coef=np.zeros((6, ))):
        """
        Initializes all of the attributes for a Model object.

        Parameters
        ----------
        base : float
            The base elevation of the aquifer [m].

        conductivity : float
            The hydraulic conductivity of the aquifer [m/d]. conductivity > 0.

        porosity : float
            The porosity of the aquifer [.]. 0 < porosity < 1.

        thickness : float
            The thickness of the aquifer [m]. thickness > 0.

        wells : list
            The list of well tuples. Each well tuple has four components.
                xw : float
                    The x-coordinate of the well [m].

                yw : float
                    The y-coordinate of the well [m].

                rw : float
                    The radius of the well [m]. rw > 0.

                qw : float
                    The discharge of the well [m^3/d].

        xo : float, optional
            The x-coordinate of the local origin [m] for the regional flow
            component of the Oneka model. Generally, this is computed during
            the model fitting process. Default is 0.

        yo : float, optional
            The y-coordinate of the local origin [m] for the regional flow
            component of the Oneka model. Generally, this is computed during
            the model fitting process. Default is 10.

        coef : ndarray, dtype=double, shape=(6, ), optional
            The six defining coefficients, A through F, for the regional flow
            component of the Oneka model. Generally, these are computed during
            the model fitting process. Default is np.zeros((6, )).

        """

        self.base = base
        self.conductivity = conductivity
        self.porosity = porosity
        self.thickness = thickness

        self.wells = wells
        self.xo = xo
        self.yo = yo
        self.coef = coef

    # ---------------------------------
    def __repr__(self):
        return 'Model({0.base}, {0.conductivity}, {0.porosity}, {0.thickness})'.format(self)

    # ---------------------------------
    def __str__(self):
        return 'Model({0.base}, {0.conductivity}, {0.porosity}, {0.thickness})'.format(self)

    # ---------------------------------
    def compute_potential(self, x, y):
        """
        Compute the discharge potential [m^3/d] at (x, y).

        Parameters
        ----------
        x : float
            The x-coordinate of the location [m].

        y : float
            The y-coordinate of the location [m].

        Returns
        -------
        potential : float
            The discharge potential [m^3/d].
        """

        # Compute the contribution from the regional flow.
        dx = x - self.xo
        dy = y - self.yo

        potential = (self.coef[0]*dx**2 + self.coef[1]*dy**2
                     + self.coef[2]*dx*dy
                     + self.coef[3]*dx + self.coef[4]*dy
                     + self.coef[5])

        # Compute the contribution from the wells.
        for well in self.wells:
            dx = x - well[0]
            dy = y - well[1]
            r2 = dx*dx + dy*dy
            potential += well[3] * np.log(r2) * 0.07957747154594767     # 1/(4*pi) = 0.07957...

        return potential

    # ---------------------------------
    def compute_discharge(self, x, y):
        """
        Compute the two components of the vertically integrated discharge
        vector [m^2/d] at (x, y).

        Parameters
        ----------
        x : float
            The x-coordinate of the location [m].

        y : float
            The y-coordinate of the location [m].

        Returns
        -------
        [Qx, Qy] : list
            The vertically integrated discharge [m^2/d].

        Notes
        -----
        o   Vertically integrated discharge is best explain by the
            Strack et al (2006). reference given below.

        References
        ----------
        o   Strack, O. D. L., Barnes, R. J., & Verruijt, A., "Vertically
            Integrated Flows, Discharge Potential, and the Dupuit-Forchheimer
            Approximation", Ground Water, 2006, 44, 72-75.
        """

        # Compute the contribution from the regional flow.
        dx = x - self.xo
        dy = y - self.yo

        Qx = -(2.0*self.coef[0]*dx + self.coef[2]*dy + self.coef[3])
        Qy = -(2.0*self.coef[1]*dy + self.coef[2]*dx + self.coef[4])

        # Compute the contribution from the wells.
        for well in self.wells:
            dx = x - well[0]
            dy = y - well[1]
            r2 = dx*dx + dy*dy

            Qx += -well[3] * dx/r2 * 0.15915494309189535    # 1/(2*pi) = 0.15915...
            Qy += -well[3] * dy/r2 * 0.15915494309189535

        return [Qx, Qy]

    # ---------------------------------
    def compute_head(self, x, y):
        """
        Compute the piezometric head [m] at (x, y).

        Parameters
        ----------
        x : float
            The x-coordinate of the location [m].

        y : float
            The y-coordinate of the location [m].

        Returns
        -------
        head : float
            The piezometric head measured from the base of the aquifer [m].

        Raises
        ------
        AquiferError
            potential <= 0
        """

        potential = self.compute_potential(x, y)

        if potential <= 0:
            raise AquiferError("potential_to_head: potential <= 0")
        elif potential < 0.5 * self.conductivity * self.thickness**2:
            head = np.sqrt(2.0 * potential / self.conductivity)
        else:
            head = ((potential + 0.5*self.conductivity*self.thickness**2)
                    / (self.conductivity*self.thickness))
        return head

    # ---------------------------------
    def compute_velocity(self, x, y):
        """
        Compute the two components of the vertically averaged seepage
        velocity vector [m/d] at (x, y).

        Parameters
        ----------
        x : float
            The x-coordinate of the location [m].

        y : float
            The y-coordinate of the location [m].

        Returns
        -------
        [Vx, Vy] : list
            The vertically averaged seepage velocity [m/d].

        Raises
        ------
        AquiferError
            head <= 0
        """

        Qx, Qy = self.compute_discharge(x, y)
        head = self.compute_head(x, y)

        if head <= 0:
            raise AquiferError("discharge_to_velocity: head <= 0")
        elif head > self.thickness:
            Vx = Qx / (self.thickness * self.porosity)
            Vy = Qy / (self.thickness * self.porosity)
        else:
            Vx = Qx / (head * self.porosity)
            Vy = Qy / (head * self.porosity)

        return (Vx, Vy)

    # ---------------------------------
    def compute_velocity_confined(self, x, y):
        """
        Compute the two components of the vertically averaged seepage
        velocity vector [m/d] at (x, y). This version of the function
        assumes that the aquifer is confined, without checking.

        Parameters
        ----------
        x : float
            The x-coordinate of the location [m].

        y : float
            The y-coordinate of the location [m].

        Returns
        -------
        [Vx, Vy] : list
            The vertically averaged seepage velocity [m^2/d].

        Raises
        ------
        None.

        Notes
        -----
        o This version of compute_velocity assumes that the aquifer is
            confined, WITHOUT checking. The justification for this kludge
            is speed. The computations are much faster because the calls
            to compute_head are unnecessary.
        """

        Qx, Qy = self.compute_discharge(x, y)
        Vx = Qx / (self.thickness * self.porosity)
        Vy = Qy / (self.thickness * self.porosity)

        return (Vx, Vy)

    # ---------------------------------
    def fit_regional_flow(self, obs, xo=np.nan, yo=np.nan):
        """
        Set the regionalflow's origin to the cetroid of the obs, then
        fit the regionalflow's coefficient using weighted least squares.

        Parameters
        ----------
        obs : list of observation tuples.
            Each observation tuple contains four values: (x, y, z_ev, z_std).
                x : float
                    The x-coordinate of the observation [m].

                y : float
                    The y-coordinate of the observation [m].

                z_ev : float
                    The expected value of the observed static water
                    level elevation [m].

                z_std : float
                    The standard deviation of the observed static water
                    level elevation [m].

        xo : float, optional
            The x-coordinate of the local origin [m] for the regional flow
            component of the Oneka model.

        yo : float, optional
            The y-coordinate of the local origin [m] for the regional flow
            component of the Oneka model.

        Returns
        -------
        An ordered pair of ndarray (coef_ev, coef_cov).
            coef_ev : ndarray, shape=(6, ), dtype=float
                The expected value vector for the model's fitted regional
                flow coefficients.

            coef_cov : ndarray, shape=(6, 6), dtype=float.
                The variance/covariance matrix for the model's fitted
                regional flow coefficients.

        Notes
        -----
        o   If xo, yo are ommitted, the centroid of the observations is
            used as the orgin of the regional flow.

        o   The caller should eliminate observations that are too
            close to pumping wells.

        o   No two observation can be at the same location. Duplicate
            observations will cause a np.linalg.LinAlgError.
        """

        # Initialize the coefficients to all 0's so that calls to
        # compute_potential during fitting include the wells only.
        self.coef = np.zeros(6, )

        # Set the local origin of the regional flow. In no origin is given
        # in the arguments, then the centroid of the observations is used.
        if np.isnan(xo) or np.isnan(yo):
            xo = np.mean([ob[0] for ob in obs])
            yo = np.mean([ob[1] for ob in obs])

        self.xo = xo
        self.yo = yo

        # Preallocate space for the fitting arrays.
        nobs = len(obs)
        A = np.zeros([nobs, 6])
        b = np.zeros([nobs, 1])
        W = np.zeros([nobs, nobs])

        # Set up the least sqaures problem.
        for i in range(nobs):
            x, y, z_ev, z_std = obs[i]

            # Compute the expected values and standard deviations of the
            # discharge potentails at the observation locations using a
            # first-order-second-moment approximation.
            head = z_ev - self.base
            if head >= self.thickness:
                pot_ev = (self.conductivity * self.thickness * (head - 0.5*self.thickness))
                pot_std = self.conductivity * self.thickness * z_std
            elif head > 0:
                pot_ev = (0.5*self.conductivity * (head**2 + z_std**2))
                pot_std = self.conductivity * head * z_std
            else:
                raise RangeError("model.fit_coeficient: elevation < base")

            A[i, :] = [(x-xo)**2, (y-yo)**2, (x-xo)*(y-yo), (x-xo), (y-yo), 1]
            b[i] = pot_ev - self.compute_potential(x, y)
            W[i, i] = 1/pot_std

        # Solve the least squares problem.
        WA = np.matmul(W, A)
        Wb = np.matmul(W, b)

        try:
            coef_ev, sse = np.linalg.lstsq(WA, Wb, rcond=-1)[0:2]
        except np.linalg.LinAlgError:
            log.error(' lstsq: failed')
            raise
        else:
            rmse = np.sqrt(sse/nobs)
            log.info(' sse = {0}, rmse = {1}'.format(sse, rmse))

        coef_ev = np.reshape(coef_ev, [6, ])

        AWWA = np.matmul(WA.T, WA)
        coef_cov = np.linalg.inv(AWWA)

        self.coef = coef_ev

        return(coef_ev, coef_cov)
