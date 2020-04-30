import cProfile
import pstats
from pstats import SortKey




# -------------------------------------
def profile_me():
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    p = pstats.Stats(pr)
    p.strip_dirs()
    p.sort_stats(SortKey.TIME).print_stats(10)


