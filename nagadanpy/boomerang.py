"""
Tools to objective/quantitatively identify influential observations
using the Kullback-Liebler divergence.

Functions
---------
boomerang(WA, Wb):
    Computes the Kullback-Leibler Divergences for leave-one-out and
    leave-two-out boomerang analyses.

compute_kldiv(mu_f, cov_f, cov_f_inv, mu_g, cov_g):
    Compute the Kullback-Leibler Divergence D_{KL}(G|F) where F and G are
    multivariate normal distributions with the same dimensions n.

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    05 May 2020
"""

import numpy as np

from nagadanpy.model import Model


# -----------------------------------------------------------------------------
def compute_boomerang(WA, Wb):
    """
    Computes the Kullback-Leibler Divergences for leave-one-out and
    leave-two-out boomerang analyses.

    Parameters
    ----------
    WA : ndarray, shape=(nobs, 6), dtype=float
        The product of the (nobs x nobs) diagonl weight matrix W times
        the (nobs x 6) regressors matrix A.

    Wb : ndarray, shape=(nobs, 1), dtype=float.
        The prodcut of the (nobs x nobs) diagonal weight matrix W times
        the (nobs x 1) response variable.

    Returns
    -------
    (kldiv_one, kldiv_two, kldiv_three) : triple of lists of tuples

        kldiv_one is a list of tuples. The tuples result from a leave-one-out
        boomerang analysis. Each tuple takes the form

            (kldiv, i)

        where i is the index (row in the obs array) of the ignored
        observation, and kl_div is the associate Kullback-Leibler
        Divergence. len(kldiv_one) = nobs.

        kldiv_two is a list of tuples. The tuples result from a leave-two-out
        boomerang analysis. Each tuple takes the form

            (kldiv, i, j)

        where i and j are the indices (rows in the obs array) of the removed
        observations, and  kl_div is the associate Kullback-Leibler Divergence.
        len(kldiv_two) = nobs*(nobs-1)/2.

    Notes
    -----
    o   The WA and Wb arguments are the return values from model.construct_fit().

    o   The reported Kullback-Leibler Divergences are computed as

            D_{KL}(G|F)

        where:
            F is the posterior multivariate normal distribution for the Oneka-type
                model parameters computed using the complete observation set.
                
            G is the posterior multivariate normal distribution for the Oneka-type
                model parameters computed using the reduced observation set.
    """

    nobs = WA.shape[0]

    mu_f, cov_f = Model.compute_fit(WA, Wb)
    cov_f_inv = np.linalg.inv(cov_f)

    kldiv_one = []
    for i in range(nobs):
        mu_g, cov_g = Model.compute_fit(np.delete(WA, i, 0), np.delete(Wb, i, 0))
        div = compute_kldiv(mu_f, cov_f, cov_f_inv, mu_g, cov_g)        
        kldiv_one.append((div, i))

    kldiv_two = []
    for i in range(nobs-1):
        for j in range(i+1, nobs):
            mu_g, cov_g = Model.compute_fit(np.delete(WA, [i, j], 0), np.delete(Wb, [i, j], 0))
            div = compute_kldiv(mu_f, cov_f, cov_f_inv, mu_g, cov_g)
            kldiv_two.append((div, i, j))

    return (kldiv_one, kldiv_two)


# -----------------------------------------------------------------------------
def compute_kldiv(mu_f, cov_f, cov_f_inv, mu_g, cov_g):
    """
    Compute the Kullback-Leibler Divergence D_{KL}(G|F) where F and G are
    multivariate normal distributions with the same dimensions n.

    Parameters
    ----------
    mu_f : ndarray, dtype=double, shape=(n, 1)
        The expected value vector for distribution F.

    cov_f : ndarray, dtype=double, shape=(n, n)
        The covariance matrix for distribution F.

    cov_f_inv : ndarray, dtype=double, shape=(n, n)
        The inverse of the covariance matrix for distribution F.

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

    a = np.trace(np.matmul(cov_f_inv, cov_g))
    b = np.matmul((mu_f - mu_g).T, np.matmul(cov_f_inv, (mu_f-mu_g)))[0,0]
    c = mu_f.shape[0]
    d = np.log(np.linalg.det(cov_f) / np.linalg.det(cov_g))

    kldiv = (a + b - c + d) / (2 * np.log(2))
    return kldiv
