"""
The entry point for the NagadanPy project.

Classes
-------
    None

Exceptions
----------
    None.

Functions
---------
    nagadan(target, npaths, duration,
            base, conductivity, porosity, thickness,
            wells, observations,
            buffer=100, spacing=10, umbra=10,
            confined=True, tol=1, maxstep=10)
        The entry-point for the NagadanPy project.

    filter_obs(observations, wells, buffer):
        Partition the obs into retained and removed. An observation is
        removed if it is within buffer of a well. Duplicate observations
        (i.e. obs at the same loction) are average using a minimum
        variance weighted average.

    log_the_run(
            target, npaths, duration,
            base, conductivity, porosity, thickness,
            wells, observations,
            buffer, spacing, umbra,
            confined, tol, maxstep)
        Print the banner and run information to the log file.

Notes
-----
o   This package is a work in progress.

o   We need to think about what events to log.

o   This module currently generates plots using python's matplotlib
    facility. We will remove these plots when we integrate into
    ArcGIS Pro.

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
    05 May 2020
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels.stats.outliers_influence as smso
import statsmodels.api as sm
import time

from nagadanpy.boomerang import compute_boomerang
from nagadanpy.capturezone import compute_capturezone
from nagadanpy.model import Model
from nagadanpy.probabilityfield import ProbabilityField
from nagadanpy.visualize import contour_head

log = logging.getLogger('NagadanPy')

VERSION = '05 May 2020'


# -----------------------------------------------
def nagadan(
        target, npaths, duration,
        base, conductivity, porosity, thickness,
        wells, observations,
        xmin=np.nan, xmax=np.nan, ymin=np.nan, ymax=np.nan,
        buffer=100, spacing=10, umbra=10,
        confined=True, tol=1, maxstep=10):
    """
    The entry-point for the NagadanPy project.

    Parameters
    ----------
    target : int
        The index identifying the target well in the wells.
        That is, the well for which we will compute a stochastic
        capture zone. This uses python's 0-based indexing.

    npaths : int
        The number of paths (starting points for the backtraces)
        to generate uniformly around the target well. 0 < npaths.

    duration : float
        The duration of the capture zone [d]. For example, a 10-year
        capture zone would have a duration = 10*365.25. 0 < duration.

    base : float
        The base elevation of the aquifer [m].

    conductivity : float
        The hydraulic conductivity of the aquifer [m/d]. 0 < conductivity.

    porosity : float
        The porosity of the aquifer []. 0 < porosity < 1.

    thickness : float
        The thickness of the aquifer [m]. 0 < thickness.

    wells : list
        The list of well tuples. Each well tuple has four components.
            xw : float
                The x-coordinate of the well [m].

            yw : float
                The y-coordinate of the well [m].

            rw : float
                The radius of the well [m]. 0 < rw.

            qw : float
                The discharge of the well [m^3/d].

    observations : list of observation tuples.
        An observation tuple contains four values: (x, y, z_ev, z_std), where
            x : float
                The x-coordinate of the observation [m].

            y : float
                The y-coordinate of the observation [m].

            z_ev : float
                The expected value of the observed static water level elevation [m].

            z_std : float
                The standard deviation of the observed static water level elevation [m].

    buffer : float, optional
        The buffer distance [m] around each well. If an obs falls
        within buffer of any well, it is removed. Default is 100 [m].

    spacing : float, optional
        The spacing of the rows and the columns [m] in the square
        ProbabilityField grids. Default is 10 [m].

    umbra : float, optional
        The vector-to-raster range [m] when mapping a particle path
        onto the ProbabilityField grids. If a grid node is within
        umbra of a particle path, it is marked as visited. Default is 10 [m].

    confined : boolean, optional
        True if it is safe to assume that the aquifer is confined
        throughout the domain of interest, False otherwise. This is a
        speed kludge. Default is True.

    tol : float, optional
        The tolerance [m] for the local error when solving the
        backtrace differential equation. This is an inherent
        parameter for an adaptive Runge-Kutta method. Default is 1.

    maxstep : float, optional
        The maximum allowed step in space [m] when solving the
        backtrace differential equation. This is a maximum space
        step and NOT a maximum time step. Default is 10.

    Returns
    -------
    None.

    Notes
    -----
    o Most of the work is orchestrated by the create_capturezone function.
    """

    # Validate the arguments.
    assert(isinstance(target, int) and 0 <= target < len(wells))
    assert(isinstance(npaths, int) and 0 < npaths)
    assert((isinstance(duration, int) or isinstance(duration, float)) and 0 < duration)

    assert(isinstance(base, int) or isinstance(base, float))
    assert((isinstance(conductivity, int) or isinstance(conductivity, float)) and 0 < conductivity)
    assert(isinstance(porosity, float) and 0 < porosity < 1)
    assert((isinstance(thickness, int) or isinstance(thickness, float)) and 0 < thickness)

    assert(isinstance(wells, list) and len(wells) >= 1)
    for we in wells:
        assert(len(we) == 4 and
               (isinstance(we[0], int) or isinstance(we[0], float)) and
               (isinstance(we[1], int) or isinstance(we[1], float)) and
               (isinstance(we[2], int) or isinstance(we[2], float)) and 0 < we[2] and
               (isinstance(we[3], int) or isinstance(we[3], float)))

    assert(isinstance(observations, list) and len(observations) > 6)
    for ob in observations:
        assert(len(ob) == 4 and
               (isinstance(ob[0], int) or isinstance(ob[0], float)) and
               (isinstance(ob[1], int) or isinstance(ob[1], float)) and
               (isinstance(ob[2], int) or isinstance(ob[2], float)) and
               (isinstance(ob[3], int) or isinstance(ob[3], float)) and 0 <= ob[3])

    assert((isinstance(buffer, int) or isinstance(buffer, float)) and 0 < buffer)
    assert((isinstance(spacing, int) or isinstance(spacing, float)) and 0 < spacing)
    assert((isinstance(umbra, int) or isinstance(umbra, float)) and 0 < umbra)

    assert(isinstance(confined, bool))
    assert((isinstance(tol, int) or isinstance(tol, float)) and 0 < tol)
    assert((isinstance(maxstep, int) or isinstance(maxstep, float)) and 0 < maxstep)

    # Initialize the stopwatch.
    start_time = time.time()

    # Log the run information.
    log_the_run(
        target, npaths, duration,
        base, conductivity, porosity, thickness,
        wells, observations,
        buffer, spacing, umbra,
        confined, tol, maxstep)

    # Filter out all of the observations that are too close to any
    # pumping well, and average the duplicate observations.
    obs = filter_obs(observations, wells, buffer)
    nobs = len(obs)
    assert(nobs > 6)

    # Set the target.
    xtarget, ytarget, rtarget = wells[target][0:3]

    # Create the model
    mo = Model(base, conductivity, porosity, thickness, wells)

    # General influence statistics
    WA, Wb = mo.construct_fit(obs, xtarget, ytarget)

    ols_model = sm.OLS(Wb, WA, hasconst=True)
    ols_results = ols_model.fit()
    ols_influence = smso.OLSInfluence(ols_results)

    log.info('\n')
    log.info(ols_results.summary())
    log.info('\n')
    log.info(ols_influence.summary_frame())
    log.info('\n')
    log.info(ols_influence.summary_table())
    
    # Compute the exhaustive leave-one-out and leave-two-out boomerang analyses.
    kldiv_one, kldiv_two = compute_boomerang(WA, Wb)

    kldiv_one.sort(reverse=True)
    kldiv_two.sort(reverse=True)

    most_influential_singleton = kldiv_one[0][1]
    most_influential_pair = [kldiv_two[0][1], kldiv_two[0][2]]

    log.info('\n')
    log.info('Top 10 of the Leave-one-out analysis:')
    for i in range(min(len(kldiv_one), 10)):
        log.info('    {0}'.format(kldiv_one[i]))

    log.info('\n')
    log.info('Top 10 of the Leave-two-out analysis:')
    for i in range(min(len(kldiv_two), 10)):
        log.info('    {0}'.format(kldiv_two[i]))

    # Define the local backtracing velocity function.
    if confined:
        def feval(xy):
            Vx, Vy = mo.compute_velocity_confined(xy[0], xy[1])
            return np.array([-Vx, -Vy])
    else:
        def feval(xy):
            Vx, Vy = mo.compute_velocity(xy[0], xy[1])
            return np.array([-Vx, -Vy])

    # Compute the three capture zones around the target well --- 
    # Using all of the obs.
    mo.fit_regional_flow(obs, xtarget, ytarget)
    pf0 = ProbabilityField(spacing, spacing, xtarget, ytarget)
    compute_capturezone(
        xtarget, ytarget, rtarget, npaths, duration,
        pf0, umbra, 1.0, tol, maxstep, feval)

    # Using all of the obs except the most influential singleton.
    obs1 = np.delete(obs, most_influential_singleton, 0)
    mo.fit_regional_flow(obs1, xtarget, ytarget)
    pf1 = ProbabilityField(spacing, spacing, xtarget, ytarget)
    compute_capturezone(
        xtarget, ytarget, rtarget, npaths, duration,
        pf1, umbra, 1.0, tol, maxstep, feval)

    # Using all of the obs except the most influential pair.
    obs2 = np.delete(obs, most_influential_pair, 0)
    mo.fit_regional_flow(obs2, xtarget, ytarget)
    pf2 = ProbabilityField(spacing, spacing, xtarget, ytarget)
    compute_capturezone(
        xtarget, ytarget, rtarget, npaths, duration,
        pf2, umbra, 1.0, tol, maxstep, feval)

    # Compute the capture zone statistics.
    Xmin = min([pf0.xmin, pf1.xmin, pf2.xmin])
    Xmax = max([pf0.xmax, pf1.xmax, pf2.xmax])
    Ymin = min([pf0.ymin, pf1.ymin, pf2.ymin])
    Ymax = max([pf0.ymax, pf1.ymax, pf2.ymax])

    pf0.expand(Xmin, Xmax, Ymin, Ymax)
    pf1.expand(Xmin, Xmax, Ymin, Ymax)
    pf2.expand(Xmin, Xmax, Ymin, Ymax)

    areaA = sum(sum(pf0.pgrid > 0)) * spacing**2
    areaB = sum(sum(pf1.pgrid > 0)) * spacing**2
    areaC = sum(sum(pf2.pgrid > 0)) * spacing**2

    areaAB = sum(sum((pf0.pgrid > 0) & (pf1.pgrid > 0))) * spacing**2
    areaAC = sum(sum((pf0.pgrid > 0) & (pf2.pgrid > 0))) * spacing**2
    areaBC = sum(sum((pf1.pgrid > 0) & (pf2.pgrid > 0))) * spacing**2

    log.info('\n')
    log.info('CAPTURE ZONE STATISTICS:')
    log.info('    A = capture zone using all observations.')
    log.info('    B = capture zone without most influenetial singleton.')
    log.info('    C = capture zone without most influenetial pair.')
    log.info('')
    log.info('    area(A)      = {0:.2f}'.format(areaA))
    log.info('    area(B)      = {0:.2f}'.format(areaB))
    log.info('    area(C)      = {0:.2f}'.format(areaC))
    log.info('')
    log.info('    area(A & B)  = {0:.2f}'.format(areaAB))
    log.info('    area(A & !B) = {0:.2f}'.format(areaA - areaAB))
    log.info('    area(B & !A) = {0:.2f}'.format(areaB - areaAB))
    log.info('')
    log.info('    area(A & C)  = {0:.2f}'.format(areaAC))
    log.info('    area(A & !C) = {0:.2f}'.format(areaA - areaAC))
    log.info('    area(C & !A) = {0:.2f}'.format(areaC - areaAC))
    log.info('')
    log.info('    area(B & C)  = {0:.2f}'.format(areaBC))
    log.info('    area(B & !C) = {0:.2f}'.format(areaB - areaBC))
    log.info('    area(C & !B) = {0:.2f}'.format(areaC - areaBC))
    log.info('')
    
    elapsedtime = time.time() - start_time
    log.info('Computational elapsed time = %.4f seconds' % elapsedtime)
    log.info('')

    # -----------------------------------------------------
    # GRAPHICAL OUTPUT STARTS HERE
    # -----------------------------------------------------

    # ---------------------------------
    # PLOT: locations of observation and wells, with contour of head.
    # ---------------------------------
    plt.figure()
    plt.axis('equal')

    plot_locations(plt, target, wells, obs)

    i = most_influential_singleton
    plt.plot(obs[i][0], obs[i][1], 'o', markeredgecolor='k',
             fillstyle='none', markersize=11)

    for i in most_influential_pair:
        plt.plot(obs[i][0], obs[i][1], 'D', markeredgecolor='k',
                 fillstyle='none', markersize=14)

    xmin, xmax, ymin, ymax = plt.axis()
    contour_head(mo, xmin, xmax, ymin, ymax, 100, 100)

    plt.xlabel('UTM Easting [m]')
    plt.ylabel('UTM Northing [m]')
    plt.title('Locations', fontsize=14)
    plt.grid(True)

    # ---------------------------------
    # PLOT: sorted KL divergence from the leave-one-out analysis.
    # ---------------------------------
    plt.figure()
    plt.scatter(range(len(kldiv_one)), [p[0] for p in kldiv_one])

    plt.xlabel('Sort Order')
    plt.ylabel('KL Divergence [bits]')
    plt.title('Leave-One-Out', fontsize=14)
    plt.grid(True)

    # ---------------------------------
    # PLOT: sorted KL divergence from the leave-two-out analysis.
    # ---------------------------------
    plt.figure()
    plt.scatter(range(len(kldiv_two)), [p[0] for p in kldiv_two])

    plt.xlabel('Sort Order')
    plt.ylabel('KL Divergence [bits]')
    plt.title('Leave-Two-Out', fontsize=14)
    plt.grid(True)

    # ---------------------------------
    # PLOT: capture zone using all observations.
    # ---------------------------------
    plt.figure()
    plt.axis('equal')

    X = np.linspace(pf0.xmin, pf0.xmax, pf0.ncols)
    Y = np.linspace(pf0.ymin, pf0.ymax, pf0.nrows)
    Z = pf0.pgrid
    plt.contourf(X, Y, Z, [0.0, 0.5, 1.0], cmap='tab10')
    plt.contour(X, Y, Z, [0.0, 0.5, 1.0], colors=['black'])

    plot_locations(plt, target, wells, obs)

    plt.xlabel('UTM Easting [m]')
    plt.ylabel('UTM Northing [m]')
    plt.title('All Data', fontsize=14)
    plt.grid(True)
    plt.axis([Xmin, Xmax, Ymin, Ymax])

    [XX, YY] = np.meshgrid(X, Y)
    XX = np.reshape(XX[Z > 0.0], -1)
    YY = np.reshape(YY[Z > 0.0], -1)

    # ---------------------------------
    # PLOT: capture zone without the most influential singleton.
    # ---------------------------------
    plt.figure()
    plt.axis('equal')

    X = np.linspace(pf1.xmin, pf1.xmax, pf1.ncols)
    Y = np.linspace(pf1.ymin, pf1.ymax, pf1.nrows)
    Z = pf1.pgrid
    plt.contourf(X, Y, Z, [0.0, 0.5, 1.0], cmap='tab10')
    plt.contour(X, Y, Z, [0.0, 0.5, 1.0], colors=['black'])

    plt.scatter(XX, YY, marker='.')
    plot_locations(plt, target, wells, obs)

    plt.xlabel('UTM Easting [m]')
    plt.ylabel('UTM Northing [m]')
    plt.title('Without Most Influential Singleton', fontsize=14)
    plt.grid(True)
    plt.axis([Xmin, Xmax, Ymin, Ymax])

    # ---------------------------------
    # PLOT: capture zone without the most influential pair.
    # ---------------------------------
    plt.figure()
    plt.axis('equal')

    X = np.linspace(pf2.xmin, pf2.xmax, pf2.ncols)
    Y = np.linspace(pf2.ymin, pf2.ymax, pf2.nrows)
    Z = pf2.pgrid
    plt.contourf(X, Y, Z, [0.0, 0.5, 1.0], cmap='tab10')
    plt.contour(X, Y, Z, [0.0, 0.5, 1.0], colors=['black'])

    plt.scatter(XX, YY, marker='.')
    plot_locations(plt, target, wells, obs)

    plt.xlabel('UTM Easting [m]')
    plt.ylabel('UTM Northing [m]')
    plt.title('Without Most Influential Pair', fontsize=14)
    plt.grid(True)
    plt.axis([Xmin, Xmax, Ymin, Ymax])
    
    # ---------------------------------
    # PLOT: Cook's distance
    # ---------------------------------
    plt.figure()
    cooks_d = ols_influence.cooks_distance[0]
    plt.bar(range(nobs), cooks_d)
    left, right = plt.xlim()
    plt.plot([left, right], [1, 1], 'r', linewidth=3)

    plt.xlabel('Observation index')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance", fontsize=14)
    plt.grid(True)

    # ---------------------------------
    # PLOT: the leverage (diagonal of the Hat matrix)
    # ---------------------------------
    plt.figure()
    leverage = ols_influence.hat_matrix_diag

    plt.bar(range(nobs), leverage)
    left, right = plt.xlim()

    threshold = 2*6/nobs
    plt.plot([left, right], [threshold, threshold], 'r', linewidth=3)

    plt.xlabel('Observation index')
    plt.ylabel('Leverage')
    plt.title('Leverage', fontsize=14)
    plt.grid(True)

    # ---------------------------------
    # PLOT: DFFITS
    # ---------------------------------
    plt.figure()
    dffits = ols_influence.dffits[0]

    plt.bar(range(nobs), dffits)
    left, right = plt.xlim()

    threshold = 2*np.sqrt(6/nobs)
    plt.plot([left, right], [threshold, threshold], 'r', linewidth=3)
    plt.plot([left, right], [-threshold, -threshold], 'r', linewidth=3)    

    plt.xlabel('Observation index')
    plt.ylabel('DFFITS')
    plt.title('DFFITS', fontsize=14)
    plt.grid(True)

    # ---------------------------------
    # PLOT: studentized residuals
    # ---------------------------------
    plt.figure()
    resid = ols_influence.resid_studentized

    plt.bar(range(nobs), resid)
    left, right = plt.xlim()

    threshold = 2
    plt.plot([left, right], [threshold, threshold], 'r', linewidth=3)
    plt.plot([left, right], [-threshold, -threshold], 'r', linewidth=3)    

    plt.xlabel('Observation index')
    plt.ylabel('Studentized Residuals')
    plt.title('Studentized Residuals', fontsize=14)
    plt.grid(True)

    # ---------------------------------
    # PLOT: normal probability plot for the studentized residuals
    # ---------------------------------
    plt.figure()
    scipy.stats.probplot(resid, fit=True, plot=plt)
    plt.ylabel('Studentized Residuals')
    plt.title('Normal Probability Plot', fontsize=14)
    plt.grid(True)
    
    plt.show()

# ------------------------------------------------------------------------------
def plot_locations(plt, target, wells, obs):

    # Plot the wells as o markers.
    xw = [we[0] for we in wells]
    yw = [we[1] for we in wells]
    plt.plot(xw, yw, 'o', markeredgecolor='k', markerfacecolor='w')

    # Plot the target well as a star marker.
    xtarget, ytarget = wells[target][0:2]
    plt.plot(xtarget, ytarget, '*', markeredgecolor='k', markerfacecolor='w', markersize=12)

    # Plot the retained observations as fat + markers.
    xo = [ob[0] for ob in obs]
    yo = [ob[1] for ob in obs]
    plt.plot(xo, yo, 'P', markeredgecolor='k', markerfacecolor='w')


# ------------------------------------------------------------------------------
def filter_obs(observations, wells, buffer):
    """
    Partition the obs into retained and removed. An observation is
    removed if it is within buffer of a well. Duplicate observations
    (i.e. obs at the same loction) are average using a minimum
    variance weighted average.

    Parameters
    ----------
    observations : list
        A list of observation tuples where the first two fields
        are x and y:
            x : float
                The x-coordinate of the observation [m].

            y : float
                The y-coordinate of the observation [m].

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
            log.info('observation removed: {0}'.format(ob))

    # Replace any duplicate observations with their weighted average.
    # Assume that the duplicate errors are statistically independent.
    obs.sort()
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
                log.info('duplicate observation: {0}'.format(obs[k]))
            retained_obs.append((obs[i][0], obs[i][1], num/den, np.sqrt(1/den)))
        else:
            retained_obs.append(obs[i])
        i = j

    log.info('')
    log.info('active observations: {0}'.format(len(retained_obs)))
    for ob in retained_obs:
        log.info('     {0}'.format(ob))
    log.info('')

    return retained_obs


# ------------------------------------------------------------------------------
def log_the_run(
        target, npaths, duration,
        base, conductivity, porosity, thickness,
        wells, observations,
        buffer, spacing, umbra,
        confined, tol, maxstep):

    log.info('\n')
    log.info('================================================================')
    log.info(' NN   N AAAAAA GGGGGG AAAAAA DDDDD  AAAAAA NN   N PPPPPP Y    Y ')
    log.info(' N N  N A    A G      A    A D    D A    A N N  N P    P  Y  Y  ')
    log.info(' N  N N AAAAAA G  GGG AAAAAA D    D AAAAAA N  N N PPPPPP   YY   ')
    log.info(' N   NN A    A G    G A    A D    D A    A N   NN P        Y    ')
    log.info(' N    N A    A GGGGGG A    A DDDDD  A    A N    N P        Y    ')
    log.info('================================================================')
    log.info('Version: {0}'.format(VERSION))
    log.info('')

    log.info('target        = {0:d}'.format(target))
    log.info('npaths        = {0:d}'.format(npaths))
    log.info('duration      = {0:.2f}'.format(duration))
    log.info('base          = {0:.2f}'.format(base))
    log.info('conductivity  = {0}'.format(conductivity))
    log.info('porosity      = {0}'.format(porosity))
    log.info('thickness     = {0:.2f}'.format(thickness))
    log.info('buffer        = {0:.2f}'.format(buffer))
    log.info('spacing       = {0:.2f}'.format(spacing))
    log.info('umbra         = {0:.2f}'.format(umbra))
    log.info('confined      = {0}'.format(confined))
    log.info('tol           = {0:.2f}'.format(tol))
    log.info('maxstep       = {0:.2f}'.format(maxstep))

    log.info('')
    log.info('wells: {0}'.format(len(wells)))
    for we in wells:
        log.info('    {0}'.format(we))

    log.info('')
    log.info('observations: {0}'.format(len(observations)))
    for ob in observations:
        log.info('    {0}'.format(ob))

    log.info('\n')
