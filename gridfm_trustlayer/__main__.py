"""`python -m gridfm_trustlayer`, the same as the `gridfm_trustlayer` command."""

import sys

from gridfm_trustlayer.cli import main

# Guarded: worker processes are spawned and import the main module again.
if __name__ == "__main__":
    sys.exit(main())
