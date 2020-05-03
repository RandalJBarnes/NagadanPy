"""
Defines and implements a single well capture zone.

Classes
-------
    None

Exceptions
----------
    Error

Functions
---------
    compute_capturezone(
            xtarget, ytarget, rtarget, npaths, duration,
            pfield, umbra, weight,
            tol, maxstep, feval)
        Compute the capture zone for the target well.

    compute_backtrace(xys, duration, tol, maxstep, feval)
        Compute a single backtrace using the Dormand-Prince adaptive
        Runge-Kutta explicit solver.

Notes
-----
o   We may be able to parallelize the tracking particles to speed up
    the overall execution.

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
    02 May 2020
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def compute_capturezone(
        xtarget, ytarget, rtarget, npaths, duration,
        pfield, umbra, weight,
        tol, maxstep, feval):
    """
    Compute the capture zone for the target well.

    Parameters
    ----------
    xtarget : float
        The x-coordinate of the target well [m].

    ytarget : float
        The y-coordinate of the target well [m].

    rtarget : float
        The radius of the target well [m]. 0 < rtarget.

    npaths : int
        The minimum number of paths (starting points for the backtraces)
        to generate uniformly around the target well. 0 < npaths.

    duration : float
        The duration of the capture zone [d]; e.g. a ten year capture zone
        will have a duration = 10*365.25. 0 < duration.

    pfield : probabilityfield
        The auto-expanding, axis-aligned, grid-based probability field.

    weight : float
        The pseudo-probability associated with the capture zone.

    umbra : float
        The vector-to-raster range [m] when mapping a particle path onto
        the ProbabilityField grids. If a grid node is within umbra of a
        particle path, it is marked as visited.

    tol : float
        The tolerance [m] for the local error when solving the backtrace
        differential equation. This is an inherent parameter for an
        adaptive Runge-Kutta method. 0 < tol.

    maxstep : float
        The maximum allowed step in space [m] when solving the backtrace
        differential equation. This is NOT the maximum time step. 0 < maxstep.

    feval : function
        The backtracing velocity function: [Vx, Vy] = feval(x, y).

    Returns
    -------
    None.

    Notes
    -----
    o   This capture zone is not stochastic.
    """

    # Local constants.
    STEPAWAY = 1.0              # Distance beyond the well radius [m].

    # Compute the backtraces.
    for theta in np.linspace(0, 2*np.pi, npaths+1)[0:-1]:
        xs = (rtarget + STEPAWAY) * np.cos(theta) + xtarget
        ys = (rtarget + STEPAWAY) * np.sin(theta) + ytarget

        vertices = compute_backtrace(xs, ys, duration, tol, maxstep, feval)
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        pfield.rasterize(x, y, umbra)

    # Register the backtraces.
    pfield.register(weight)


# -------------------------------------
def compute_backtrace(xs, ys, duration, tol, maxstep, feval):
    """
    Compute a single backtrace using the Dormand-Prince adaptive
    Runge-Kutta explicit solver.

    Parameters
    ----------
    xs : float
        The x-coordinate [m] of the starting point.

    ys : float
        The y-coordinate [m] of the starting point.

    duration : float
        The duration of the capture zone [d]; e.g. a ten year capture zone
        will have a duration = 10*365.25. 0 < duration.

    tol : float
        The tolerance [m] for the local error when solving the backtrace
        differential equation. This is an inherent parameter for an
        adaptive Runge-Kutta method. 0 < tol.

    maxstep : float
        The maximum allowed step in space [m] when solving the backtrace
        differential equation. 0 < maxstep.

    feval : function
        The backtracing velocity function: [Vx, Vy] = feval(x, y).

    Returns
    -------
    vertices : list
        The list of (x, y) tuples containing the vertices of the
        backtrace path.

    Notes
    -----
    o   This function use the Dormand-Prince implementation of an
        adaptive Runge-Kutta algorithm to solve the backtrace ode.
        See, for example,

            https://en.wikipedia.org/wiki/Dormand-Prince_method

    o   Much of the Dormand-Prince Runge-Kutta code is translated
        and adpated from the MATLAB implementation given by
        www.mathtools.com.

    o   This implementation is unique is two significant ways:

        --  The feval is NOT a function of time, as we are working
            with steadystate flow.

        --  The maximum time step is governed by a maximum space step.

    o   The Dormand-Prince Runge-Kutta is currently the default method
        in MATLAB's ode45, as well as SciPy's ode integration library.

    References
    ----------
    o   Dormand, J. R.; Prince, P. J. (1980), A family of embedded
        Runge-Kutta formulae, Journal of Computational and Applied
        Mathematics, 6 (1): 19–26, doi:10.1016/0771-050X(80)90013-3.

    o   Dormand, John R. (1996), Numerical Methods for Differential
        Equations: A Computational Approach, Boca Raton: CRC Press,
        pp. 82–84, ISBN 0-8493-9433-3.

    o   https://www.mathstools.com/section/main/dormand_prince_method
    """

    # Local constants.
    EPS = np.finfo(float).eps

    # Dormand-Prince constants for an adaptive Runge-Kutta integrator.
    a2 = np.array([1/5])
    a3 = np.array([3/40, 9/40])
    a4 = np.array([44/45, -56/15, 32/9])
    a5 = np.array([19372/6561, -25360/2187, 64448/6561, -212/729])
    a6 = np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656])
    a7 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])

    e = np.array([71/57600, -1/40, -71/16695, 71/1920, -17253/339200, 22/525])

    # Initialize: the starting point is the first vertex.
    t = 0
    dt = 0.1                # This is an arbitrary choice.

    vertices = [(xs, ys)]
    xy = np.array([xs, ys])

    try:
        k1 = feval(xy)

        while (t < duration):
            # Do not step past duration.
            if (t + dt > duration):
                dt = duration - t

            # The 5th order Runge-Kutta with Dormand-Prince constants.
            k2 = feval(xy + dt*(a2[0]*k1))
            k3 = feval(xy + dt*(a3[0]*k1 + a3[1]*k2))
            k4 = feval(xy + dt*(a4[0]*k1 + a4[1]*k2 + a4[2]*k3))
            k5 = feval(xy + dt*(a5[0]*k1 + a5[1]*k2 + a5[2]*k3 + a5[3]*k4))
            k6 = feval(xy + dt*(a6[0]*k1 + a6[1]*k2 + a6[2]*k3 + a6[3]*k4 + a6[4]*k5))

            xyt = xy + dt*(a7[0]*k1 + a7[2]*k3 + a7[3]*k4 + a7[4]*k5 + a7[5]*k6)

            # Control time step for maximum space step and maximum estimated error.
            k2 = feval(xyt)
            est = np.linalg.norm(dt*(e[0]*k1 + e[1]*k2 + e[2]*k3 + e[3]*k4 +
                                     e[4]*k5 + e[5]*k6), np.inf)
            ds = np.linalg.norm(xyt - xy)

            if (est < tol) and (ds < maxstep):
                t = t + dt
                k1 = k2
                xy = xyt
                vertices.append(xy)

            dt = 0.9 * min((tol/(est + EPS))**(1/5), maxstep/(ds + EPS), 10) * dt

    except:
        log.warning(' trace terminated prematurely at t = {0:.2f} < duration.'.format(t))

    finally:
        return vertices
