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
    02 May 2020
"""

PROJECTNAME = 'Carlos example'

TARGET = 0
NPATHS = 500
DURATION = 10*365.25

BASE = 0.0
CONDUCTIVITY = 15.2
POROSITY = 0.25
THICKNESS = 25

BUFFER = 100
SPACING = 10
UMBRA = 20

CONFINED = True
TOL = 1
MAXSTEP = 20

WELLS = [(322579, 5093431, 0.2, 179)]

OBSERVATIONS = [
    (324125, 5094758.0, 413.0040, 1.6),
    (322332, 5093693.0, 411.7848, 1.6),
    (322636, 5094024.0, 410.8704, 1.6),
    (323438, 5092221.0, 419.1000, 1.6),
    (322474, 5092682.0, 415.7472, 1.6),
    (323402, 5092650.0, 420.0000, 1.5),
    (324334, 5092974.0, 423.0000, 1.5),
    (324201, 5092476.0, 421.0000, 1.5),
    (321612, 5092354.0, 418.0000, 1.5),
    (321739, 5093159.0, 414.0000, 1.5),
    (321820, 5093298.0, 414.0000, 1.5),
    (323384, 5094353.0, 415.0000, 1.5),
    (323060, 5091983.0, 423.0000, 1.5)
    ]
