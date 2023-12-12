
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models import Token
from app.services.authentication import authenticate_user, create_access_token
from datetime import timedelta
import os
from sqlalchemy.orm import Session
from app.logging_config import setup_logger  # Logging setup
from app.database import get_db
import yaml

import pathlib
#from app.logging_config import setup_logger


current_path = pathlib.Path(__file__).parent

config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
# Environment and DB setup
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

router = APIRouter()
logger = setup_logger()
# Retrieve configuration from environment variables
SECRET_KEY = config.get("SECRET_KEY")
ALGORITHM = config.get("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(config.get("ACCESS_TOKEN_EXPIRE_MINUTES"))

# FastAPI route to handle login and token issuance
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Generates a JWT access token for a user upon successful login.
    
    Parameters:
    - form_data: An instance of OAuth2PasswordRequestForm containing the login form data.
    - db: An instance of the database Session.
    
    Returns:
    - A dictionary containing the access token and token type.
    """
    # Authenticate the user based on the form data
    user = authenticate_user(db, form_data.username, form_data.password)
    
    # If authentication fails, log the event and raise an HTTP Exception
    if not user:
        logger.warning(f"login_for_access_token: Invalid credentials for user {form_data.username}")  # Log warning
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Calculate token expiration time
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Generate the access token
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Log successful login
    logger.info(f"login_for_access_token: valid credentials for user {form_data.username}")  
    
    # Return the access token
    return {"username": user.username, "access_token": access_token, "token_type": "bearer"}
