from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File, Security
from app.models import VectorDBRequest,LoginUser
from app.depencencies.security import get_current_active_user
from app.depencencies.security import get_current_active_user
from app.database import User
import requests
import os
import yaml
import secrets
import zipfile
from typing import Optional
import pathlib
from sqlalchemy.orm import Session
from app.database import get_db
from pydantic import BaseModel, parse_raw_as
import weaviate
import logging
import json

# Load environment variables and setup
current_path = pathlib.Path(__file__).parent
config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
received_files_dir = os.path.join(current_path.parent.parent.parent, 'received_files')
os.makedirs(received_files_dir, exist_ok=True)

logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

logger = logging.getLogger(__name__)
logger.propagate = True

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

#weaviate_client = weaviate.Client("http://localhost:8080")

Ray_service_URL = config.get("Ray_service_URL")
router = APIRouter()


@router.post("/")
async def query_vectorDB( class_name: str = Form(...),
                          mode: str = Form(...),
                          vectorDB_type: str = Form("Weaviate"),
                          file_path: Optional[str] = Form(None),
                          current_user: User = Depends(get_current_active_user),
                          file: Optional[UploadFile] = File(None)):
    try:
        print(f"received data: {class_name}, {mode}, {vectorDB_type}, {file_path}, {current_user.username}")
        print(f"Received file: {file.filename if file else 'No file'}")
        username = current_user.username

        # Check if a file is included in the request
        if file:
            # Create a random directory for the file
            random_dir = secrets.token_hex(8)  # Generates a random 16-character string
            file_dir = os.path.join(received_files_dir, random_dir)
            os.makedirs(file_dir, exist_ok=True)
            # Save the file in the random directory
            file_path = os.path.join(file_dir, file.filename)
            with open(file_path, "wb") as file_object:
                file_object.write(await file.read())
            # If the file is a ZIP file, extract it
            if file.content_type == 'application/zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract only the files not in _MACOSX directory
                    for zip_info in zip_ref.infolist():
                        if not zip_info.filename.startswith('__MACOSX/'):
                            zip_ref.extract(zip_info, file_dir)
            # Update the file path in the request data
            file_path = file_dir

        data_dict = {
            "username": username,
            "class_name": class_name,
            "mode": mode,
            "vectorDB_type": vectorDB_type,
            "file_path": file_path
        }
        # Send the request to the external service
        response = requests.post(f"{Ray_service_URL}/VectorDB", json=data_dict)
        print('response', response) 
        response.raise_for_status()  # Raises an HTTPError for unsuccessful status codes
        response_data = response.json()
        return {"username": current_user.username, "response": response_data}
    except requests.HTTPError as e:
        if response.status_code == 400:
            raise HTTPException(status_code=400, detail="Bad request to the other API service.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to forward request to the other API service. Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")