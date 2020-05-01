"""
Computes the Kullback-Leibler Divergences for leave-one-out and leave-two-out
boomerang analyses.

Functions
---------
boomerang(mo, obs):
    Implements leave-one-out and leave-two-out boomerang analyses.

Notes
-----

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    01 May 2020
"""

import numpy as np


# -----------------------------------------------------------------------------
def compute_boomerang(mo, obs):
    """
    Computes the Kullback-Leibler Divergences for leave-one-out and
    leave-two-out boomerang analyses.

    Parameters
    ----------
    mo : onekapy.model.Model
        The well-formed oneka-type model.

    obs : list of observation tuples.
        Each observation tuple contains four values: (x, y, z_ev, z_std).
            x : float
                The x-coordinate of the observation [m].

            y : float
                The y-coordinate of the observation [m].

            z_ev : float
                The expected value of the observed static water level 
                elevation [m].

            z_std : float
                The standard deviation of the observed static water level 
                elevation [m]. 0 < z_std.

    Returns
    -------
    [kldiv_one, kldiv_two] : pair of lists of tuples
    
        kldiv_one is a list of tuples. The tuples result from a leave-one-out 
        boomerang analysis. Each tuple takes the form

            (kldiv, i)

        where i is the index (row in the obs array) of the ignored 
        observation, and kl_div is the associate Kullback-Leibler 
        Divergence. len(kldiv_one) = nobs.

        kldiv_two is a list of tuples. The tuples result from a leave-two-out 
        boomerang analysis. Each tuple take thes form

            (kldiv, i, j)

        where i and j are the indices (rows in the obs array) of the removed
        observations, and  kl_div is the associate Kullback-Leibler Divergence.
        len(kldiv_two) = nobs*(nobs-1)/2.

        The rows kldiv_one and kldiv_two arrays are sorted on the kldiv, from
        highest to lowest.

    Notes
    -----
    o   The reported Kullback-Leibler Divergences are computed as

            D_{KL}(G|F)

        where:
            F is the posterior multivariate normal distribution for the Oneka-type
                model parameters computed using the complete observation set.
            G is the posterior multivariate normal distribution for the Oneka-type
                model parameters computed using the reduced observation set.

    o   The centroid of the complete observation set is used as the origin of
        the Oneka-type model.
    """
    xo = np.mean([ob[0] for ob in obs])
    yo = np.mean([ob[1] for ob in obs])

    mu_f, cov_f = mo.fit_regional_flow(obs, xo, yo)
    nobs = len(obs)

    kldiv_one = []
    for i in range(nobs):
        ob = np.delete(obs, i, 0)
        mu_g, cov_g = mo.fit_regional_flow(ob, xo, yo)
        kl_div = compute_kldiv(mu_f, cov_f, mu_g, cov_g)
        kldiv_one.append((kl_div, i))
    kldiv_one.sort(reverse=True)

    kldiv_two = []
    for i in range(nobs-1):
        for j in range(i+1, nobs):
            ob = np.delete(obs, [i, j], 0)
            mu_g, cov_g = mo.fit_regional_flow(ob, xo, yo)
            kl_div = compute_kldiv(mu_f, cov_f, mu_g, cov_g)
            kldiv_two.append((kl_div, i, j))
    kldiv_two.sort(reverse=True)

    return (kldiv_one, kldiv_two)


# -----------------------------------------------------------------------------
def compute_kldiv(mu_f, cov_f, mu_g, cov_g):
    """
    Compute the Kullback-Leibler Divergence D_{KL}(G|F) where F and G are
    multivariate normal distributions with the same dimensions n.

    Parameters
    ----------
    mu_f : ndarray, dtype=double, shape=(n, 1)
        The expected value vector for distribution F.
    cov_f : ndarray, dtype=double, shape=(n, n)
        The covariance matrix for distribution F.
    mu_g : ndarray, dtype=double, shape=(n, 1)
        The expected value vector for distribution G.
    cov_g : ndarray, dtype=double, shape=(n, n)
        The covariance matrix for distribution G.

    Returns
    -------
    kldiv : double
        Kullback-Leibler Divergence D_{KL}(G|F). The units are [bits].

    Notes
    -----
    o   See <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>
    """
    cov_f_inv = np.linalg.inv(cov_f)

    a = np.trace(np.matmul(cov_f_inv, cov_g))
    b = np.matmul((mu_f - mu_g).T, np.matmul(cov_f_inv, (mu_f-mu_g)))
    c = mu_f.shape[0]
    d = np.log(np.linalg.det(cov_f) / np.linalg.det(cov_g))

    kldiv = (a + b - c + d) / (2 * np.log(2))
    return kldiv
