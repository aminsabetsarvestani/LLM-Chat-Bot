from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File, Security
from app.models import VectorDBRequest,LoginUser, ArxivInput
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
from pydantic import BaseModel
import weaviate
import logging
import json

# Load environment variables and setup
current_path = pathlib.Path(__file__).parent
config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
received_files_dir = os.path.join(current_path.parent.parent.parent, 'received_files')
os.makedirs(received_files_dir, exist_ok=True)

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

logger = logging.getLogger(__name__)
logger.propagate = True

#weaviate_client = weaviate.Client("http://localhost:8080")

Ray_service_URL = config.get("Ray_service_URL")
router = APIRouter()

@router.post("/")
async def query_arxiv_search(class_name: str = Form(None),
                            query: str = Form(None),
                            paper_limit: int = Form(None),
                            recursive_mode: int = Form(None),
                            mode: str = Form(...),
                            title: str = Form(None),
                            url: str = Form(None),
                            file_path: Optional[str] = Form(None),
                            dir_name: Optional[str] = Form(None),
                            current_user: User = Depends(get_current_active_user),
                            file: Optional[UploadFile] = File(None)):
    print('checkpoint api 1')
    print('data received 1', 'cls name', class_name, 'mode', mode, 'query', query, 'recursive mode', recursive_mode, 'paper limit', paper_limit, 'file path', file_path, 'ursername', current_user.username, 'title', title, 'url', url)
    username = current_user.username
    try:
        print('data received', class_name, mode, query, recursive_mode, paper_limit, file_path, current_user.username)
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
            file_path = file_dir
        if url:
            random_dir = secrets.token_hex(8)  # Generates a random 16-character string
            file_dir = os.path.join(received_files_dir, random_dir)
            os.makedirs(file_dir, exist_ok=True)
            dir_name = file_dir
            print('dir name', dir_name)

        data_dict = {
            "username": username,
            "class_name": class_name,
            "query": query,
            "paper_limit": paper_limit,
            "recursive_mode": recursive_mode,
            "mode": mode,
            "title": title,
            "url": url,            
            "file_path": file_path,
            "dir_name": dir_name
        }
        
        #print(f"Received data: {vector_db_request.dict()}")  # Debug print
        response = requests.post(f"{Ray_service_URL}/ArxivSearch/", json=data_dict)
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