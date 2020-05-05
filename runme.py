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
    03 May 2020
"""

from datetime import datetime
import importlib
import logging
import os
import time

import cProfile
import pstats
from pstats import SortKey

from nagadanpy.nagadan import nagadan


# ------------------------------------------------------------------------------
def main(module_name):

    try:
        m = importlib.import_module(module_name)
        print('Running: {0}'.format(m.PROJECTNAME))
    except ModuleNotFoundError:
        print('ModuleNotFoundError: No module named {0}'.format(module_name))
        return

    # Initialize the run.
    start_time = time.time()

    if not os.path.exists('logs'):
        os.makedirs('logs')    

    logging.basicConfig(
        filename='logs\\NagadanPy' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.log',
        filemode='w',
        level=logging.INFO)
    log = logging.getLogger(__name__)

    log.info(' Project: {0}'.format(m.PROJECTNAME))
    log.info(' Run date: {0}'.format(time.asctime()))

    # Call the working function.
    nagadan(
        m.TARGET, m.NPATHS, m.DURATION,
        m.BASE, m.CONDUCTIVITY, m.POROSITY, m.THICKNESS,
        m.WELLS, m.OBSERVATIONS,
        m.BUFFER, m.SPACING, m.UMBRA,
        m.CONFINED, m.TOL, m.MAXSTEP)

    # Shutdown the run.
    elapsedtime = time.time() - start_time
    log.info('Total elapsed time = %.4f seconds' % elapsedtime)
    logging.shutdown()

    print('\nTotal elapsed time = %.4f seconds' % elapsedtime)


# ------------------------------------------------------------------------------
def profile_me(module_name):
    pr = cProfile.Profile()
    pr.enable()
    main(module_name)
    pr.disable()

    p = pstats.Stats(pr)
    p.strip_dirs()
    p.sort_stats(SortKey.TIME).print_stats(10)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    module_name = input('Enter the module name (without .py): ')
    main(module_name)
