from langchain_community.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from app.routers.LLM.backend_database import Database
import json
my_data = Database()

my_data.update_conversation( {
                    "username": "admin",
                    "content": "HI",
                    "gen_token_number": 1,
                    "prompt_token_number": 2,
                    "conversation_number": 3,
                })

