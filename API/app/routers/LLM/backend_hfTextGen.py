from langchain.chains.conversation.memory import ConversationBufferMemory

import json
import re
from typing import Optional
from pydantic import BaseModel
import textwrap
from langchain.chains import RetrievalQA
import logging
from langchain_community.vectorstores import Weaviate
import weaviate
import wandb
from app.routers.LLM.backend_database import Database
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain import LLMChain
from typing import List
import json
import yaml
import time
import asyncio
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import pathlib
import os
import transformers
from torch import cuda, bfloat16
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.vectorstores import Weaviate
from app.logging_config import setup_logger
from langchain import hub

# ------------------- Configuration --------------------
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)



class Input(BaseModel):
    username: Optional[str]
    prompt: Optional[str]
    memory: Optional[bool]
    conversation_number: Optional[int]
    AI_assistance: Optional[bool]
    collection_name: Optional[str]
    llm_model: Optional[str]


# ------------------------------ LLM Deployment -------------------------------
class LLMDeployment:
    def __init__(self, model_tokenizer,
                 temperature =  0.01,
                 max_new_tokens=  512,
                 repetition_penalty= 1.1,
                 batch_size= 2):
        

        self.logger = setup_logger()

        current_path = pathlib.Path(__file__).parent.parent.parent.parent
        config_path = current_path/ 'cluster_conf.yaml'

        # Environment variables setup
        with open(config_path, 'r') as self.file:
            self.config = yaml.safe_load(self.file)
            self.config = Config(**self.config)
        # Initialize Weaviate client for RAG
        try:
            self.weaviate_client = weaviate.Client(
                url=self.config.weaviate_client_url,   
            )
        except:
            self.logger.error("Error in connecting to Weaviate")
            self.RAG_enabled = False

        #setting up weight and bias logging
        self.wandb_logging_enabled = self.config.WANDB_ENABLE
        if self.wandb_logging_enabled:
            try:
                wandb.login(key = self.config.WANDB_KEY)
                wandb.init(project="Service Metrics", notes="custom step")
                wandb.define_metric("The number of input tokens")
                wandb.define_metric("The number of generated tokens")
                wandb.define_metric("Inference Time")
                wandb.define_metric("token/second")
                self.logger.info("Wandb Logging Enabled")
            except:
                self.wandb_logging_enabled = False
                self.logger.info("Wandb Logging Not Enabled")
                pass
        
        # Initialize deployment class
        self.model_tokenizer = model_tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.batch_size = batch_size
        self.loop = asyncio.get_running_loop()
        self.access_token = self.config.Hugging_ACCESS_TOKEN
        self.device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # initialize Prompt 
        
        prompt_hub = hub.pull("llmchatbot/default",api_key = self.config.Langchain_access_key, api_url="https://api.hub.langchain.com")
        
        
        self.DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        self.instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
      
        

        self.template = self.get_prompt(self.instruction)
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"], template=self.template
        )
        self.prompt = prompt_hub
        # initialize tokenizer for recording data related to requests e.g. number of input/ouput tokens
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_tokenizer, padding_side="left", token =self.access_token
        )
        # Initialize LLM
        self.llm = HuggingFaceTextGenInference(
    inference_server_url=self.config.LLMs[0]["url"],
    max_new_tokens=self.config.LLMs[0]["max_new_tokens"],
    top_k=self.config.LLMs[0]["top_k"],
    top_p=self.config.LLMs[0]["top_p"],
    typical_p=self.config.LLMs[0]["typical_p"],
    temperature=self.config.LLMs[0]["temperature"],
    repetition_penalty=self.config.LLMs[0]["repetition_penalty"],
    streaming=self.config.LLMs[0]["streaming"],
)
       
        # Initialize Memory

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output"
        )


        # Initialize RAG if enabled
        if self.RAG_enabled:
            self.weaviate_vectorstore = Weaviate(
                self.weaviate_client, 
                "Admin_General_collection",
                'page_content', 
                attributes=['page_content']
            )
            
            self.QA_document = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.weaviate_vectorstore.as_retriever(),
                memory=self.memory,
                output_key="output",
            )

        # Initialize Database to store/restore chat history
            
        self.database = Database()

    def get_collection_based_retriver(self, client, collection):
        content = ['page_content']
        self.logger.info("collection is %s", collection)
        weaviate_vectorstore = Weaviate(
            client,
            str(collection),
            'page_content', 
            attributes=['page_content']
        )
        return weaviate_vectorstore
    

    def get_prompt(self, instruction):
        SYSTEM_PROMPT = self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS
        prompt_template = self.B_INST + SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template

    def cut_off_text(self, text, prompt):
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    def remove_substring(self, string, substring):
        return string.replace(substring, "")

    def cleaning_memory(self):
        print(self.memory.chat_memory.messages)
        self.memory.clear()
        print("Chat History Deleted")

    def parse_text(self, text):
        pattern = r"\s*Assistant:\s*"
        pattern2 = r"\s*AI:\s*"
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = re.sub(pattern2, "", cleaned_text)
        wrapped_text = textwrap.fill(cleaned_text, width=100)
        return wrapped_text

    def InferenceCall(self, request: Input):
        try:
            self.logger.info("Received request by backend: %s", request.dict())
            input_prompt = request.prompt
            AI_assistance = request.AI_assistance
            username = request.username
            memory = request.memory
            conversation_number = request.conversation_number
            collection_name = request.collection_name
            
            if not memory:
                memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True, output_key="output"
                )
            else:
                # Retrieve the latest conversation from the database
                chat_history = None
                if conversation_number <= 0 or conversation_number is None:
                    latest_chat = self.database.retrieve_conversation(
                        {
                            "username": username,
                        }
                    )
                    if "content"  in latest_chat:
                        chat_history = latest_chat["content"]
                        conversation_number = latest_chat["conversation_number"]
                        logging.info(
                            f" the latest conversation for {username} with the conversation_id of {conversation_number} retrieved from database"
                        )
                else:
                    chat_history = self.database.retrieve_conversation(
                        {
                            "username": username,
                            "conversation_number": conversation_number,
                        }
                    )
                    chat_history = chat_history["content"]
                    logging.info(
                        f"chat histroy for {username} with the conversation_number of {conversation_number} retrieved from database"
                    )

                # Initialize memory from the retrieved conversation
                if chat_history:
                    retrieved_messages = messages_from_dict(json.loads(chat_history))
                    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
                    memory = ConversationBufferMemory(
                        chat_memory=retrieved_chat_history, memory_key="chat_history"
                    )
                else: 
                    memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True, output_key="output"
                )

            # Create an LLM chain with the appropriate mode

            if AI_assistance:
                llm_chain =LLMChain(llm=self.llm, prompt=self.prompt  , verbose=False, memory=memory, output_key="output")
            else:
                #check if collection exists
                if self.database.check_collection_exists(request.dict()):
                    if username[0].isalpha():
                        new_username= username[0].upper() + username[1:]
                    collection_name = f"{new_username}_{collection_name}"
                    retriever = self.get_collection_based_retriver(self.weaviate_client,collection_name)
                    
                    llm_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=retriever.as_retriever(),
                        memory=memory,
                        output_key="output",
                        
                    )
                else: 
                    return {"output": "Error: Collection does not exist"}
            pre_inference_memo = llm_chain.memory.chat_memory.messages
            pre_inference_memo = " ".join(
                message.content for message in pre_inference_memo
            )
            pre_inference_memo_token_len = len(
                self.tokenizer.tokenize(pre_inference_memo)
            )
            # Generate a response based on the mode
            inference_start_time = time.time()
            

            # Generate a response based on the mode
            if AI_assistance:
                response = llm_chain({"user_input": input_prompt})["output"]
            else:
                logging.info("input_prompt is %s", input_prompt)
                response = llm_chain.run(input_prompt)
                logging.info("response from TGI is %s", response)


            # End measuring time after inference
            inference_end_time = time.time()

            # Calculate and log the elapsed time
            inference_elapsed_time = inference_end_time - inference_start_time

            # Store the conversation
            extracted_messages = llm_chain.memory.chat_memory.messages
            ingest_to_db = messages_to_dict(extracted_messages)
            input_token_number = (
                len(self.tokenizer.tokenize(input_prompt))
                + pre_inference_memo_token_len
            )
            gen_token_number = len(self.tokenizer.tokenize(response))
            
            db_response = self.database.update_conversation(
    {
        "username": username,
        "content": json.dumps(ingest_to_db),
        "gen_token_number": gen_token_number,
        "prompt_token_number": input_token_number,
        "conversation_number": conversation_number,
    }
)
            response = self.parse_text(response)

            wandb_log = {
                "The number of input tokens": input_token_number,
                "The number of generated tokens": gen_token_number,
                "Inference Time": inference_elapsed_time,
                "token/second": gen_token_number / inference_elapsed_time,
            }
            if self.wandb_logging_enabled:   
                wandb.log(wandb_log)
            self.logger.info("Processed the request successfully")
            return {"output": response}

        except ConnectionError as ce:
            self.logger.error("Error processing the request: %s", str(ce))
            # Handle connection errors (for example, interacting with the database or calling APIs)
            return {"output": "ConnectionError: An error occurred while processing the request"}

        except KeyError as ke:
            # Handle key errors (for example, accessing a key in a dictionary that doesn’t exist)
            self.logger.error("Error processing the request: %s", str(ke))
            return {"output": " KeyError: An error occurred while processing the request"}

        except Exception as e:
            # General exception to catch any other unforeseen errors
            self.logger.error("Error processing the request: %s", str(e))
            return {"output": "Exception : An error occurred while processing the request"}
        

