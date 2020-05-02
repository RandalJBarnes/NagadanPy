"""
Defines and implements an auto-expanding, axis-aligned-grid-based
probability field.

Classes
-------
    ProbabilityField

Raises
------
    RangeError

Notes
-----
o   TODO: Add some explanation of what the ProbabilityField is,
    and how it works.

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
import math
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------
class Error(Exception):
    """Base class for module errors."""
    pass


# ---------------------------------------------------------
class RangeError(Error):
    """Passed argument out of valid range."""
    pass


# ---------------------------------------------------------
class ProbabilityField:
    """
    An auto-expanding, axis-aligned-grid-based probability field.

    Attributes
    ----------
    deltax : float
        The spacing of grid columns [m].

    deltay : float
        The spacing of grid rows [m].

    nrows : integer
        number of rows in the grids.

    ncols : integer
        number of colums in the grids.

    xmin : float
        The minimum x-coordinate [m] of the axis-aligned grids.

    xmax : float
        The maximum x-coordinate [m] of the axis-aligned grids.

    ymin : float
        The minimum y-coordinate [m] of the axis-aligned grids.

    ymax : float
        The maximum y-coordinate [m] of the axis-aligned grids.

    total_weight : float
        The total weight (pseudo-probability) of all registered realization.
        This is the normalizing value.

    pgrid : ndarray, dtype=np.float, shape=(nrows, ncols)
        The accumulated pseudo-proability grid.

    rgrid = ndarray, dtype=np.bool, shape=(nrows, ncols)
        The boolean registration array for the current realization.

    Methods
    -------
    expand(self, xmin, xmax, ymin, ymax):
        If necessary, expand the grids to include xmin, xmax, ymin, and ymax.
        The contents of the grids are maintained.

    rasterize(self, x, y, umbra):
        Insert a single track into the registration array (rgrid), using a
        vector-to-raster range of umbra.

    insert(self, ax, ay, bx, by, umbra):
        Insert a single linear segment [(ax, ay), (bx, by)] of a single track
        into the registration array (rgrid), using a vector-to-raster range
        of umbra.

    register(self, weight):
        Add the current realization's registration rgrid into the current
        probability pgrid with pseudo-probability weight. The registration
        rgrid is then reset.

    reset(self):
        Reset registration rgrid. This is needed when discarding a
        partially processed invalid realization.

    @staticmethod
    distancesquared(ax, ay, bx, by, cx, cy):
        Computes the distance squared between the point (cx,cy) and the line
        segment defined by [(ax,ay),(bx,by)].  Note, this is the distance to
        the line segment, not the distance to the line.
    """

    # ---------------------------------
    def __init__(self, deltax, deltay, xo=np.nan, yo=np.nan):
        # validate the parameters
        if deltax <= 0:
            raise RangeError("<deltax> must be > 0.")

        if deltay <= 0:
            raise RangeError("<deltay> must be > 0.")

        self.deltax = deltax
        self.deltay = deltay

        if np.isnan(xo) or np.isnan(yo):
            self.nrows = 0
            self.ncols = 0
        else:
            self.xmin = xo - deltax
            self.xmax = xo + deltax
            self.ymin = yo - deltay
            self.ymax = yo + deltay

            self.nrows = 3
            self.ncols = 3

            self.pgrid = np.zeros((self.nrows, self.ncols), dtype=np.float)
            self.rgrid = np.zeros((self.nrows, self.ncols), dtype=np.bool)

            self.total_weight = 0.0

    # ---------------------------------
    def __repr__(self):
        if (self.nrows == 0):
            return ("ProbabilityField({0.deltax}, {0.deltay})"
                    .format(self))
        else:
            return ("ProbabilityField({0.deltax}, {0.deltay}, {0.nrows}, {0.ncols}, "
                    "{0.xmin}, {0.xmax}, {0.ymin}, {0.ymax}, {0.total_weight})"
                    .format(self))

    # ---------------------------------
    def __str__(self):
        if (self.nrows == 0):
            return ("ProbabilityField({0.deltax}, {0.deltay}, {0.nrows}, {0.ncols})"
                    .format(self))
        else:
            return ("ProbabilityField({0.deltax}, {0.deltay}, {0.nrows}, {0.ncols}, "
                    "{0.xmin}, {0.xmax}, {0.ymin}, {0.ymax}, {0.total_weight}, "
                    "\n{0.pgrid!r}, \n{0.rgrid!r})"
                    .format(self))

    # ---------------------------------
    def expand(self, xmin, xmax, ymin, ymax):
        """
        If necessary, expand the grids to include xmin, xmax, ymin, and ymax.
        The contents of the grids are maintained.

        Parameters
        ----------
        xmin : float
            trial minimum x-coordinate [m].
        xmax : float
            trial maximum x-coordinate [m].
        ymin : float
            trial minimum y-coordinate [m].
        ymax : float
            trial maximum y-coordinate [m].

        Returns
        -------
        None.
        """

        # Validate the arguments.
        if xmin > xmax:
            raise RangeError("<xmin> must be <= <xmax>.")

        if ymin > ymax:
            raise RangeError("<ymin> must be <= <ymax>.")

        # Intialize the grids so that (xmin, ymin) is up one row and over one column
        # (e.g. in index [1,1], not [0,0]) from the lower left corner.
        if (self.ncols == 0) or (self.nrows == 0):
            self.xmin = xmin - self.deltax
            self.ymin = ymin - self.deltay

            self.ncols = max(3, math.ceil((xmax - self.xmin)/self.deltax) + 2)
            self.nrows = max(3, math.ceil((ymax - self.ymin)/self.deltay) + 2)

            self.xmax = self.xmin + (self.ncols-1)*self.deltax
            self.ymax = self.ymin + (self.nrows-1)*self.deltay

            self.pgrid = np.zeros((self.nrows, self.ncols), dtype=np.float)
            self.rgrid = np.zeros((self.nrows, self.ncols), dtype=np.bool)

            self.total_weight = 0.0

            log.debug(' grids expanded to ({self.nrows}, {self.ncols}).')
        else:
            # Determine the new grid geometry.
            nrows = self.nrows
            ncols = self.ncols

            rshift = 0
            cshift = 0

            while xmin <= self.xmin:
                self.xmin -= self.deltax
                ncols += 1
                cshift += 1

            while xmax >= self.xmax:
                self.xmax += self.deltax
                ncols += 1

            while ymin <= self.ymin:
                self.ymin -= self.deltay
                nrows += 1
                rshift += 1

            while ymax >= self.ymax:
                self.ymax += self.deltay
                nrows += 1

            # Create the expanded grids, if necessary.
            if (nrows != self.nrows) or (ncols != self.ncols):
                pgrid = np.zeros((nrows, ncols), dtype=np.float)
                rgrid = np.zeros((nrows, ncols), dtype=np.bool)

                rgrid[rshift:rshift+self.nrows, cshift:cshift+self.ncols] = self.rgrid
                pgrid[rshift:rshift+self.nrows, cshift:cshift+self.ncols] = self.pgrid

                self.rgrid = rgrid
                self.pgrid = pgrid

                self.nrows = nrows
                self.ncols = ncols

                log.debug(' grids expanded to ({0}, {1})'.format(nrows, ncols))

    # ---------------------------------
    def insert(self, ax, ay, bx, by, umbra):
        """
        Insert a single linear segment [(ax, ay), (bx, by)] of a single track into the
        probability field registration array (rgrid), using a vector-to-raster umbra.

        Parameters
        ----------
        ax : float
            The x-coordinate [m] of line segment end point a.

        ay : float
            The y-coordinate [m] of line segment end point a.

        bx : float
            The x-coordinate [m] of line segment end point b.

        by : float
            The y-coordinate [m] of line segment end point b.

        umbra : float
            The vector-to-raster range [m]. Think of the umbra as half the
            width of the line segment.

        Returns
        -------
        None.
        """

        # Make certain that the grid is large enough.
        # self.expand(min(ax, bx), max(ax, bx), min(ay, by), max(ay, by))

        # Determine the sub-grid limits.
        umbra_squared = umbra*umbra

        left = max(0, math.floor((min(ax, bx) - umbra - self.xmin)/self.deltax))
        right = min(self.ncols, math.floor((max(ax, bx) + umbra - self.xmin)/self.deltax) + 1)
        bottom = max(0, math.floor((min(ay, by) - umbra - self.ymin)/self.deltay))
        top = min(self.nrows, math.floor((max(ay, by) + umbra - self.ymin)/self.deltay) + 1)

        for j in np.arange(left, right):
            for i in np.arange(bottom, top):
                if not(self.rgrid[i, j]):
                    cx = self.xmin + j*self.deltax
                    cy = self.ymin + i*self.deltay

                    if self.distancesquared(ax, ay, bx, by, cx, cy) < umbra_squared:
                        self.rgrid[i, j] = True

    # ---------------------------------
    def rasterize(self, x, y, umbra):
        """
        Insert a single track into the registration array (rgrid), using a
        vector-to-raster range of width.

        Parameters
        ----------
        x : ndarray, dtype=double, shape=(n, ).
            Array of x-coordinates [m] of the particle track.

        y : ndarray, dtype=double, shape=(n, ).
            Array of x-coordinates [m] of the particle track.

        umbra : float
            The vector-to-raster range [m]. Think of the umbra as half the
            width of the line segment.

        Returns
        -------
        None.
        """
        # Make certain that the grid is large enough.
        self.expand(min(x), max(x), min(y), max(y))

        # Insert each line segment.
        for i in range(len(x)-1):
            self.insert(x[i], y[i], x[i+1], y[i+1], umbra)

    # ---------------------------------
    def register(self, weight):
        """
        Add the current realization's registration rgrid into the current probability
        pgrid with pseudo-probability weight. The registration rgrid is then reset.

        Parameters
        ----------
        weight : float
            Pseudo-probability associated with the current realization.

        Returns
        -------
        None.
        """

        self.total_weight += weight
        self.pgrid[self.rgrid] += weight
        self.rgrid[:] = False

    # ---------------------------------
    def reset(self):
        """
        Reset registration rgrid. This is needed when discarding a
        partially processed invalid realization.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        self.rgrid[:] = False

    # ---------------------------------
    @staticmethod
    def distancesquared(ax, ay, bx, by, cx, cy):
        """
        Computes the distance squared between the point (cx,cy) and the line
        segment defined by [(ax,ay),(bx,by)].  Note, this is the distance to
        the line segment, not the distance to the line.

        Parameters
        ----------
        ax : float
            x-coordinate of endpoint a.
        ay : float
            y-coordinate of endpoint a.
        bx : float
            x-coordinate of endpoint b.
        by : float
            y-coordinate of endpoint a.
        cx : float
            x-coordinate of point c.
        cy : float
            y-coordinate of point c.

        Returns
        -------
        d2 : float
            distance squared between point c and the line segment [a,b].
        """

        bax = bx - ax
        bay = by - ay

        cax = cx - ax
        cay = cy - ay

        perpdot = bax*cay - bay*cax
        dot = bax*cax + bay*cay
        length2 = bax*bax + bay*bay

        alpha2 = perpdot*perpdot / length2
        beta2 = dot*dot / length2

        if(dot < 0):
            d2 = alpha2 + beta2
        elif(beta2 > length2):
            d2 = alpha2 + beta2 - 2*dot + length2
        else:
            d2 = alpha2

        return d2
