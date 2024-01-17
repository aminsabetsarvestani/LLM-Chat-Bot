import requests
from typing import TypedDict

class InferenceRequest(TypedDict):
    prompt: str
    memory: bool
    conversation_number: int
    AI_assistance: bool
    collection_name: str
    llm_model: str 




class llmService:
    def __init__(self, base_url, access_token: str = None):
        self.base_url = base_url

        if access_token:
            self.access_token = access_token
        else:
            self.access_token = None
    
    def query_inference(self, query_data: InferenceRequest):
        if not self.access_token:
            raise Exception("Not authenticated, please set you access token using authenticate() method")
        headers = {"Authorization": f"Bearer {self.access_token}"}
        resp = requests.post(f"{self.base_url}/llm_request/", json=query_data, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception("Inference failed")
        
    def retrieve_latest_conversation(self):
        if not self.access_token:
            raise Exception("Not authenticated, please set you access token using authenticate() method")
        headers = {"Authorization": f"Bearer {self.access_token}"}
        resp = requests.post(f"{self.base_url}/de_request/retrieve_latest_conversation/", headers=headers, json={})
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception("Database request failed")
        