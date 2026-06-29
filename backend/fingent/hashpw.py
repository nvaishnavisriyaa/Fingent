"""Mint a PBKDF2 password hash for use in FINGENT_USERS.

Usage:
    python -m fingent.hashpw 'my-strong-password'
Then put the printed hash in FINGENT_USERS, e.g.:
    FINGENT_USERS='{"alice": {"password": "<hash>", "tenant": "acme", "roles": ["admin"]}}'
"""
import sys

from .auth import hash_password


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m fingent.hashpw <password>", file=sys.stderr)
        return 2
    print(hash_password(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
