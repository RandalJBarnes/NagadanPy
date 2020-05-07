"""
A small set of utilitiy functions.

Functions
---------
    contour_head(mo, xmin, xmax, ymin, ymax, nrows, ncols):

    contour_potential(mo, xmin, xmax, ymin, ymax, nrows, ncols)

    filter_obs(observations, wells, buffer):
        Partition the obs into retained and removed. An observation is
        removed if it is within buffer of a well. Duplicate observations
        (i.e. obs at the same loction) are average using a minimum
        variance weighted average.

    quick_capture_zone(mo, we, nrays, nyears, maxstep, fmt)

    summary_statistics(values, names, formats, title)
        Create a simple summary statistics table.

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    07 May 2020
"""

import io
import logging
import matplotlib.pyplot as plt
import numpy as np

import nagadanpy.model

log = logging.getLogger('NagadanPy')


# ------------------------------------------------------------------------------
def contour_head(mo, xmin, xmax, ymin, ymax, nrows, ncols):
    x = np.linspace(xmin, xmax, ncols)
    y = np.linspace(ymin, ymax, nrows)

    grid = np.zeros((nrows, ncols), dtype=np.double)

    for i in range(nrows):
        for j in range(ncols):
            try:
                grid[i, j] = mo.compute_head(x[j], y[i])
            except nagadanpy.model.AquiferError:
                grid[i, j] = np.nan

    plt.contourf(x, y, grid, cmap='bwr')
    plt.colorbar()


# ------------------------------------------------------------------------------
def contour_potential(mo, xmin, xmax, ymin, ymax, nrows, ncols):
    x = np.linspace(xmin, xmax, ncols)
    y = np.linspace(ymin, ymax, nrows)

    grid = np.zeros((nrows, ncols), dtype=np.double)

    for i in range(nrows):
        for j in range(ncols):
            try:
                grid[i, j] = mo.compute_potential(x[j], y[i])
            except nagadanpy.model.AquiferError:
                grid[i, j] = np.nan

    plt.contourf(x, y, grid, cmap='bwr')
    plt.colorbar()


# ------------------------------------------------------------------------------
def filter_obs(observations, wells, buffer):
    """
    Partition the obs into retained and removed. An observation is
    removed if it is within buffer of a well. Duplicate observations
    (i.e. obs at the same loction) are average using a minimum
    variance weighted average.

    Arguments
    ---------
    observations : list
        An observation tuple contains four values: (x, y, z_ev, z_std), where
            x : float
                The x-coordinate of the observation [m].

            y : float
                The y-coordinate of the observation [m].

            z_ev : float
                The expected value of the observed static water level 
                elevation [m].

            z_std : float
                The standard deviation of the observed static water level 
                elevation [m].

    wells : list
        A list of well tuples where the first two fields of the
        tuples are xw and yw:
            xw : float
                The x-coordinate of the well [m].

            yw : float
                The y-coordinate of the well [m].

        Note: the well tuples may have other fields, but the first
        two must be xw and yw.

    buffer : float
        The buffer distance [m] around each well. If an obs falls
        within buffer of any well, it is removed.

    Returns
    -------
    retained_obs : list
        A list of the retained observations. The fields are the
        same as those in obs. These include averaged duplicates.

    Notes
    -----
    o   Duplicate observations are averaged and the associated
        standard deviation is updated to reflect this.

    o   We use a weighted average, with the weight for the i'th
        obs proportional to 1/sigma^2_i. This is the minimum
        variance estimator. See, for example,
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    """

    # Local constants.
    TOO_CLOSE = 1.0             # Minimum distance for unique obs.

    # Remove all observations that are too close to pumping wells.
    obs = []
    for ob in observations:
        flag = True
        for we in wells:
            if np.hypot(ob[0]-we[0], ob[1]-we[1]) <= buffer:
                flag = False
                break
        if flag:
            obs.append(ob)
        else:
            log.info('observation removed: {0} is too close to {1}'.format(ob, we))

    # Replace any duplicate observations with their weighted average.
    # Assume that the duplicate errors are statistically independent.
    retained_obs = []

    i = 0
    while i < len(obs):
        j = i+1
        while (j < len(obs)) and (np.hypot(obs[i][0]-obs[j][0], obs[i][1]-obs[j][1]) < TOO_CLOSE):
            j += 1

        if j-i > 1:
            num = 0
            den = 0
            for k in range(i, j):
                num += obs[k][2]/obs[k][3]**2
                den += 1/obs[k][3]**2
                log.info('    duplicate observation: {0}'.format(obs[k]))
            retained_obs.append((obs[i][0], obs[i][1], num/den, np.sqrt(1/den)))
            log.info('averaged observation created: {0}'.format(retained_obs[-1]))
        else:
            retained_obs.append(obs[i])
        i = j

    log.info('')
    log.info('active observations: {0}'.format(len(retained_obs)))
    for i in range(len(retained_obs)):
        log.info('     [{0:03d}] {1}'.format(i, retained_obs[i]))

    log.info('')
    log.info('Note: the index numbers associated with these active')
    log.info('      observations may differ from the original index')
    log.info('      numbers. This is a consequence of averaging ')
    log.info('      duplicates and deleting observations near wells.')
    log.info('      These new index numbers are used in the subsequent')
    log.info('      reporting. Be aware!')

    return retained_obs


# ------------------------------------------------------------------------------
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

        except nagadanpy.model.AquiferError:
            print(f"Aquifer error (e.g. dry) for theta = {theta:.3f}")
            continue


# ------------------------------------------------------------------------------
def summary_statistics(values, names, formats, title):
    """
    Creates a simple summary statistics table.

    Arguments
    ---------
    values : array-like
        The array of data values.

    names: list of strings
        The list of variable names.

    formats : list of strings
        The list of format strings.

    title : string
        The title string.

    Returns
    -------
    buf : io.StringIO
        Use buf.getvalue() to access the string.

    Usage
    -----
    buf = summary_statistics(data, ['X', 'Y', 'Z'], ['8.2f', '6.3f', '9.4e'], 'This is a TITLE')    
    
    """

    # Compute sizes of stuff.
    nvar = len(values[0])
    widths = [len('{0:{fmt}}'.format(1, fmt=formats[j])) for j in range(nvar)]
    total_width = 5 + sum(widths)

    # Compute the summary statistics.
    vcnt = [[] for j in range(nvar)]
    vmin = [[] for j in range(nvar)]
    vmed = [[] for j in range(nvar)]
    vmax = [[] for j in range(nvar)]
    vavg = [[] for j in range(nvar)]
    vstd = [[] for j in range(nvar)]

    for j in range(nvar):
        x = np.array([v[j] for v in values])
        x = x[~np.isnan(x)]

        vcnt[j] = len(x)
        vmin[j] = np.min(x)
        vmed[j] = np.median(x)
        vmax[j] = np.max(x)
        vavg[j] = np.mean(x)
        vstd[j] = np.std(x)

    # Initialize.
    buf = io.StringIO()

    # Write out the header.
    buf.write('{0:^{w}s}'.format(title, w=total_width))
    buf.write('\n')    
    buf.write('=' * total_width)
    buf.write('\n')    

    buf.write('     ')
    for j in range(nvar):
        buf.write('{0:>{w}s}'.format(names[j], w=widths[j]))
    buf.write('\n')
    buf.write('-' * total_width)
    buf.write('\n')

    # Write out the cnt  
    buf.write('cnt: ')
    for j in range(nvar):
        buf.write('{0:>{w}d}'.format(vcnt[j], w=widths[j]))
    buf.write('\n')

    # Write out the min
    buf.write('min: ')
    for j in range(nvar):
        buf.write('{0:{fmt}}'.format(vmin[j], fmt=formats[j]))
    buf.write('\n')

    # Write out the med
    buf.write('med: ')
    for j in range(nvar):
        buf.write('{0:{fmt}}'.format(vmed[j], fmt=formats[j]))
    buf.write('\n')

    # Write out the max 
    buf.write('max: ')
    for j in range(nvar):
        buf.write('{0:{fmt}}'.format(vmax[j], fmt=formats[j]))
    buf.write('\n')

    # Write out the avg
    buf.write('avg: ')
    for j in range(nvar):
        buf.write('{0:{fmt}}'.format(vavg[j], fmt=formats[j]))
    buf.write('\n')

    # Write out the std
    buf.write('std: ')
    for j in range(nvar):
        buf.write('{0:{fmt}}'.format(vstd[j], fmt=formats[j]))
    buf.write('\n')

    # Write out the footer.
    buf.write('=' * total_width)
    buf.write('\n')

    return buf
