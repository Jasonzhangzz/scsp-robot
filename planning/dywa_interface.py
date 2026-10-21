"""Compatibility import; implementation lives in :mod:`planning.dywa.runtime`."""
import sys
from planning.dywa import runtime

if __name__ == "__main__":
    runtime.main()
else:
    sys.modules[__name__] = runtime
