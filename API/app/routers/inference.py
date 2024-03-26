from fastapi import APIRouter, Depends, HTTPException
from app.models import  InferenceRequest
from app.depencencies.security import get_current_active_user
from app.database import User
import requests
import yaml
import pathlib
import yaml
import logging
from app.routers.LLM.backend_hfTextGen import LLMDeployment
from app.logging_config import setup_logger
current_path = pathlib.Path(__file__)

config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
# Environment and DB setup
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

logger = setup_logger()
def get_route_prefix_for_llm(llm_name):
    for llm in config['LLMs']:
        if llm['name'] == llm_name:
            return llm['route_prefix']
    return None

Ray_service_URL = config.get("Ray_service_URL")
llm = LLMDeployment(model_tokenizer="meta-llama/Llama-2-13b-chat-hf")
router = APIRouter()

@router.post("/")
async def create_inference(data: InferenceRequest, current_user: User = Depends(get_current_active_user)):
    logger.info("Received request by router: %s", data.dict())
    try:
        #data.memory = False
        data.username = current_user.username
        response = llm.InferenceCall(data)
        return {"username": current_user.username, "data":response}

    except requests.HTTPError as e:
        if response.status_code == 400:
            raise HTTPException(status_code=400, detail="Bad request to the other API service.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to forward request to the other API service. Error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
