"""Policy Iteration Lab backend."""

import os

# The tabular models are small; multi-threaded BLAS only adds overhead (and badly oversubscribes on
# some MKL builds). Must run before numpy is imported anywhere in the package.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _load_env_local() -> None:
    """Local dev convenience: read KEY=VALUE lines from .env.local (never used on Vercel)."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / ".env.local"
    if os.environ.get("VERCEL") or not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value)


if not os.environ.get("PI_LAB_IGNORE_ENV_LOCAL"):
    _load_env_local()
