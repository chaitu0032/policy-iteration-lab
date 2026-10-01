"""Policy Iteration Lab backend."""

import os

# The tabular models are small; multi-threaded BLAS only adds overhead (and badly oversubscribes on
# some MKL builds). Must run before numpy is imported anywhere in the package.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
