import os

from fastapi import Request
from fastapi.security import HTTPBasic


async def get_credentials(request: Request):
    """
    Determines whether to apply optional security measures based on the value of the AUTH_API environment variable.

    Args:
        request (Request): The incoming request object.

    Returns:
        Union[HTTPBasic, None]: An instance of the HTTPBasic class if AUTH_API is set to "True", otherwise None.
    """
    if os.getenv("AUTH_API") == "True":
        return await HTTPBasic(request)
    else:
        return None
