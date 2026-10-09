"""Password hashing and JWT helpers."""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings

# PBKDF2-SHA256 is a vetted KDF with no native dependency. (passlib 1.7.4's
# bcrypt backend is incompatible with bcrypt>=5, so we avoid it.)
_pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """Return a salted hash of ``password`` suitable for storage."""
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check ``plain_password`` against a stored hash."""
    return _pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT whose ``sub`` claim is the user's id."""
    settings = get_settings()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    claims = {"sub": str(user_id), "exp": expire}
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    """Validate ``token`` and return the user id, or ``None`` if invalid/expired."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None
