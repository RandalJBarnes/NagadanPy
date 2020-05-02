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

from datetime import datetime
import logging
import time

from nagadanpy.nagadan import nagadan


# ======================================
# Here are the necessary data.
# ======================================
PROJECTNAME = 'Barnesville example'

TARGET = 0
NPATHS = 500
DURATION = 10*365.25

BASE = 0.0
CONDUCTIVITY = 3.2
POROSITY = 0.25
THICKNESS = 100

BUFFER = 100
SPACING = 10
UMBRA = 20

CONFINED = True
TOL = 1
MAXSTEP = 20

WELLS = [(0.0, 0.0, 0.2, 312.0)]

OBSERVATIONS = [
    (-857, -737, 306.02, 1.62),
    (-1128, 686, 309.07, 1.62),
    (799, -1218, 318.21, 1.62),
    (-1081, 1867, 306.48, 1.62),
    (-2333, 756, 298.7, 1.62),
    (-2833, -602, 301.45, 1.62),
    (3243, -107, 332.84, 1.62),
    (-2823, -1971, 302.06, 1.62),
    (-515, 3679, 306.63, 1.62),
    (-722, 4128, 309.37, 1.62),
    (-3677, -2236, 300.23, 1.62),
    (805, -4232, 315.47, 1.62),
    (2244, 4038, 331.62, 1.62),
    (-1091, 4552, 303.58, 1.62),
    (-3127, -3688, 301.14, 1.62),
    (1357, -4784, 321.26, 1.62),
    (3340, -4336, 333.45, 1.62),
    (5617, 184, 338.33, 1.62),
    (4415, 3641, 336.19, 1.62),
    (6239, 423, 342.6, 1.62),
    (1157, -6174, 320.65, 1.62),
    (5493, 3338, 343.81, 1.62),
    (-1815, -6173, 306.93, 1.62),
    (-3184, -5778, 301.75, 1.62),
    (5993, 3198, 341.68, 1.62),
    (5902, -3423, 342.29, 1.62),
    (7262, 1039, 344.42, 1.62),
    (4196, 6022, 335.28, 1.62),
    (1359, 7822, 328.27, 1.62),
    (6332, -6215, 345.95, 1.62),
    (-666, 9303, 317.6, 1.62)
    ]

# ======================================
# Here is the form of the base call.
# ======================================
if __name__ == "__main__":
    # execute only if run as a script

    # Initialize the run.
    start_time = time.time()

    logging.basicConfig(
        filename='NagadanPy' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.log',
        filemode='w',
        level=logging.INFO)
    log = logging.getLogger(__name__)

    log.info(' Project: {0}'.format(PROJECTNAME))
    log.info(' Run date: {0}'.format(time.asctime()))

    # Call the working function.
    nagadan(
        TARGET, NPATHS, DURATION,
        BASE, CONDUCTIVITY, POROSITY, THICKNESS,
        WELLS, OBSERVATIONS,
        BUFFER, SPACING, UMBRA,
        CONFINED, TOL, MAXSTEP)

    # Shutdown the run.
    elapsedtime = time.time() - start_time
    log.info('Total elapsed time = %.4f seconds' % elapsedtime)
    logging.shutdown()

    print('\n\nTotal elapsed time = %.4f seconds' % elapsedtime)
