"""
A small set of model-based utilitiy functions. These are not
part of the OnekaPy package.

Functions
---------
    contour_head(mo, xmin, xmax, ymin, ymax, nrows, ncols):

    contour_potential(mo, xmin, xmax, ymin, ymax, nrows, ncols)

    quick_capture_zone(mo, we, nrays, nyears, maxstep, fmt)

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    21 April 2020
"""

import matplotlib.pyplot as plt
import numpy as np

import aquifer


# -------------------------------------
def contour_head(mo, xmin, xmax, ymin, ymax, nrows, ncols):
    x = np.linspace(xmin, xmax, ncols)
    y = np.linspace(ymin, ymax, nrows)

    grid = np.zeros((nrows, ncols), dtype=np.double)

    for i in range(nrows):
        for j in range(ncols):
            try:
                grid[i, j] = mo.compute_head(np.array([x[j], y[i]]))
            except aquifer.AquiferError:
                grid[i, j] = np.nan

    plt.contourf(x, y, grid, cmap='bwr')
    plt.colorbar()


# -------------------------------------
def contour_potential(mo, xmin, xmax, ymin, ymax, nrows, ncols):
    x = np.linspace(xmin, xmax, ncols)
    y = np.linspace(ymin, ymax, nrows)

    grid = np.zeros((nrows, ncols), dtype=np.double)

    for i in range(nrows):
        for j in range(ncols):
            try:
                grid[i, j] = mo.compute_potential(np.array([x[j], y[i]]))
            except aquifer.AquiferError:
                grid[i, j] = np.nan

    plt.contourf(x, y, grid, cmap='bwr')
    plt.colorbar()


# ---------------------------------
def quick_capture_zone(mo, we, nrays, nyears, maxstep, fmt):
    """
    Compute and plot a capture zone for Well we using Model mo.

    Parameters
    ----------
    mo : Model
        The driving model for the capture zone.

    we : Well
        The well for which the capture zone is computed

    nrays : int
        The number of uniformly distributed rays to trace out from the well.

    nyears : double
        The number years to run the back trace.

    maxstep : float
        The solve_ivp max_step parameter.

    fmt : string
        The format string for the backtrace plot.

    Returns
    -------
    None.
    """

    radius = we.radius + 1
    xc = we.x
    yc = we.y

    for theta in np.linspace(0, 2*np.pi, nrays):
        xo = radius*np.cos(theta) + xc
        yo = radius*np.sin(theta) + yc

        try:
            sol = mo.compute_backtrack(xo, yo, nyears*365, maxstep)

            for year in np.arange(0, nyears):
                idx = np.logical_and(year*365 < sol.t, sol.t < (year+1)*365)

                if (year % 2) == 0:
                    plt.plot(sol.y[0, idx], sol.y[1, idx], fmt, linewidth=4)
                else:
                    plt.plot(sol.y[0, idx], sol.y[1, idx], '-k')

        except aquifer.AquiferError:
            print(f"Aquifer error (e.g. dry) for theta = {theta:.3f}")
            continue
