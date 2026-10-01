"""Keep unit tests independent of a developer's .env.local accounts."""

import os

os.environ["PI_LAB_IGNORE_ENV_LOCAL"] = "1"
os.environ.pop("PI_LAB_USERS", None)
os.environ.pop("PI_LAB_SECRET", None)
