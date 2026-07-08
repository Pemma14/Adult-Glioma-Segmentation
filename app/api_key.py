import logging

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.auth.API_KEY:
        logger.warning("Unauthorized access attempt with key ending in ...%s", x_api_key[-4:])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return True
