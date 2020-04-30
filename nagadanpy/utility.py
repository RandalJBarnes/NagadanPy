"""
Implements some simple unitility functions.

Classes
-------
    None.

Functions
---------
    isnumber(arg)
    isposnumber(arg)
    isposint(arg)
    isvalidindex(arg, n)

Notes
-----


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
    30 April 2020
"""


def isnumber(arg):
    return isinstance(arg, int) or isinstance(arg, float)


def isint(arg):
    return isinstance(arg, int)
