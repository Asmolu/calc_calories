from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core import security


@dataclass
class User:
    id: str
    email: str
    hashed_password: str
    full_name: Optional[str] = None


def create_demo_user(email: str = "demo@calai.app", password: str = "demo") -> User:
    """Create a deterministic demo user instance."""

    return User(id="demo", email=email, hashed_password=security.hash_password(password))


def verify_demo_user(email: str, password: str) -> bool:
    """Validate demo credentials."""

    user = create_demo_user()
    return user.email == email and security.verify_password(password, user.hashed_password)