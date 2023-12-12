# filename: authentication.py

from datetime import datetime, timedelta
import os
from jose import jwt
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from passlib.context import CryptContext
from app.logging_config import setup_logger
from app.database import User

import yaml

import pathlib
#from app.logging_config import setup_logger
import yaml

current_path = pathlib.Path(__file__).parent

config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Fetch configuration from environment variables
SECRET_KEY = config.get("SECRET_KEY")
ALGORITHM = config.get("ALGORITHM")
DB_SERVICE_URL = config.get("DB_SERVICE_URL")
# Initialize logger
logger = setup_logger()
# Password hashing utility
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Verify if provided password matches the hashed password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies if a plain password matches a hashed password.

    Args:
        plain_password (str): The plain password to be verified.
        hashed_password (str): The hashed password to compare against.

    Returns:
        bool: True if the plain password matches the hashed password, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

# Generate a hashed password from plain text
def get_password_hash(password: str) -> str:
    """
    Generate a password hash using the provided password.

    Parameters:
        password (str): The password to be hashed.

    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(password)

# Authenticate a user by verifying username and password
def authenticate_user(db: Session, username: str, password: str) -> User:
    """
    Authenticates a user by checking the provided username and password against the database.

    Parameters:
        db (Session): The database session.
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        User: The authenticated user if the username and password are valid, otherwise False.
    """
    try:
        user = db.query(User).filter(User.username == username).first()
    except SQLAlchemyError as e:
        logger.error(f"Database error during authentication: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="Internal server error") from e
    if user is None or not verify_password(password, user.hashed_password):
        return False
    return user

# Generate an access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Creates an access token for the given data (usename) and optional expiration delta.

    Parameters:
        data (dict): The data to be encoded into the access token.
        expires_delta (timedelta, optional): The optional expiration delta for the access token. Defaults to None.

    Returns:
        str: The encoded access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
