"""


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
    04 May 2020
"""

# ======================================
# Here are the necessary data.
# ======================================
PROJECTNAME = 'Basic example'

TARGET = 0
NPATHS = 25
DURATION = 10*365.25

BASE = 0.0
CONDUCTIVITY = 20.0
POROSITY = 0.25
THICKNESS = 15.0

BUFFER = 100
SPACING = 10
UMBRA = 10

CONFINED = True
TOL = 1
MAXSTEP = 20

WELLS = [
    (2250, 2250, 0.25, 750),
    (1750, 2750, 0.25, 750)
    ]

OBSERVATIONS = [
    (1000, 1000, 100, 2),
    (1000, 1500, 105, 2),
    (1000, 2000, 110, 2),
    (1000, 2500, 115, 2),
    (1000, 3000, 120, 2),
    (1500, 1000, 95, 2),
    (1500, 1500, 100, 2),
    (1500, 2000, 105, 2),
    (1500, 2500, 110, 2),
    (1500, 3000, 115, 2),
    (2000, 1000, 90, 2),
    (2000, 1500, 95, 2),
    (2000, 2000, 100, 2),
    (2000, 2500, 105, 2),
    (2000, 3000, 110, 2),
    (2500, 1000, 85, 2),
    (2500, 1500, 90, 2),
    (2500, 2000, 95, 2),
    (2500, 2500, 100, 2),
    (2500, 3000, 105, 2),
    (3000, 1000, 80, 2),
    (3000, 1500, 85, 2),
    (3000, 2000, 90, 2),
    (3000, 2500, 95, 2),
    (3000, 3000, 100, 2)
    ]
