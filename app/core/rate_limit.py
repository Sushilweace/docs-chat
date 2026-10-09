"""slowapi rate limiter keyed by authenticated user (falling back to client IP)."""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.security import decode_access_token


def rate_limit_key(request: Request) -> str:
    """Identify the caller: ``user:<id>`` from a valid bearer token, else the IP.

    Keying by user rather than IP alone means users behind a shared NAT don't
    throttle each other, and one user can't dodge the limit by changing IP.
    """
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token:
        user_id = decode_access_token(token)
        if user_id is not None:
            return f"user:{user_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key)
