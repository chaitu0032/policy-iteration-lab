"""Generate the lab's accounts: strong random passwords, stored only as salted PBKDF2 hashes.

    python3 scripts/create_users.py                 # user1..user5
    python3 scripts/create_users.py alice bob ...   # custom names

Writes (both git-ignored):
  .env.local              PI_LAB_USERS + PI_LAB_SECRET  -> loaded by the local server;
                          copy both values into Vercel > Project > Settings > Environment Variables
  credentials.local.txt   the plain passwords, to hand out (delete it afterwards)
"""

from __future__ import annotations

import os
import secrets
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.auth import USERNAME_PATTERN, hash_password  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ALPHABET = string.ascii_letters + string.digits
PASSWORD_LENGTH = 16


def _write_private(path: Path, text: str) -> None:
    """Create/truncate with mode 0600 from the first byte (no window with default permissions)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o600)


def main(names: list[str]) -> None:
    names = names or [f"user{i}" for i in range(1, 6)]
    bad = [n for n in names if not USERNAME_PATTERN.fullmatch(n)]
    if bad:
        sys.exit(f"Invalid username(s): {', '.join(bad)} (use a-z, 0-9, _ or -, 2-32 chars)")
    creds = {n: "".join(secrets.choice(ALPHABET) for _ in range(PASSWORD_LENGTH)) for n in names}
    users = ",".join(f"{n}:{hash_password(p)}" for n, p in creds.items())
    secret = secrets.token_urlsafe(48)
    _write_private(ROOT / ".env.local", f"PI_LAB_USERS={users}\nPI_LAB_SECRET={secret}\n")
    _write_private(ROOT / "credentials.local.txt",
                   "Policy Iteration Lab accounts (keep private, delete after handing out)\n\n"
                   + "\n".join(f"{n:10s} {p}" for n, p in creds.items()) + "\n")
    print(f"Created {len(names)} accounts: {', '.join(names)}")
    print("Passwords -> credentials.local.txt ; env vars -> .env.local (both git-ignored)")


if __name__ == "__main__":
    main(sys.argv[1:])
