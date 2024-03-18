# filename: authentication.py

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.models import TokenData
from app.database import get_db, User
from datetime import timedelta, datetime
import os
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import logging
import yaml

# Load environment variables


import pathlib
#from app.logging_config import setup_logger
import yaml

current_path = pathlib.Path(__file__)

config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
# Environment and DB setup
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read environment variables
SECRET_KEY = config.get("SECRET_KEY")
ALGORITHM = config.get("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(config.get("ACCESS_TOKEN_EXPIRE_MINUTES"))

# Initialize OAuth2 with token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Async function to get the current user based on the token
async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    """
    Retrieves the current user based on the provided token.

    Parameters:
        token (str): The authentication token.
        db (Session): The database session.

    Returns:
        User: The user object associated with the provided token.

    Raises:
        HTTPException: If the token is invalid or the credentials are not valid.
    """
    # Define an exception for invalid credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the token using the SECRET_KEY and ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        # If username is not found, raise an exception
        if username is None:
            raise credentials_exception
    except JWTError:
        logger.warning("JWT validation failed")
        raise credentials_exception

    # Query the user from the database
    user = db.query(User).filter(User.username == username).first()

    # If the user doesn't exist, raise an exception
    if user is None:
        raise credentials_exception

    return user


# Async function to get the current active user
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """
    Check if the user account is disabled.

    Parameters:
        current_user (User): The current user object.

    Returns:
        User: The current active user.

    Raises:
        HTTPException: If the user account is disabled.
    """
    # Check if the user account is disabled
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user
