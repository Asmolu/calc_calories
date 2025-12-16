import hashlib
from typing import Final


_SALT: Final[bytes] = b"calai-demo"


def hash_password(raw_password: str) -> str:
    """Return a deterministic hash for a password.

    This lightweight helper is not intended to replace a production-grade
    password hasher but provides a simple mechanism for demos and tests.
    """

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        _SALT,
        100_000,
    )
    return digest.hex()


def verify_password(raw_password: str, hashed_password: str) -> bool:
    """Validate that a password matches a stored hash."""

    return hash_password(raw_password) == hashed_password