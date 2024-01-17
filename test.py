import os
import asyncio
from typing import Any
from typing import AsyncIterable
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.chains import LLMChain
app = FastAPI()
class Message(BaseModel):
    content: str
# initialize the agent (we need to do this for the callbacks)
from langchain_community.llms import HuggingFaceTextGenInference
from langchain import PromptTemplate
from langchain.chains import LLMChain

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"

prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}
Human: {input}
AI:"""

def get_prompt( instruction):
        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        return prompt_template
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8082/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    streaming=True,
)
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# LLM



template = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \n\nUser: {question}"
prompt = PromptTemplate(
            input_variables=[ "question"], template=template
        )

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name

conversation = LLMChain(llm=llm, prompt=prompt, verbose=False, output_key="output")

# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory


async def run_call(content: str,callback: AsyncIteratorCallbackHandler):
    # assign callback handler
    conversation.callbacks = [callback]
    # now query
    await conversation.acall(inputs={"question": content})



async def send_message(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    task = asyncio.create_task(run_call(content,callback))
    print("task", task)
    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()
    await task


@app.post("/stream_chat/")
async def stream_chat(message: Message):
    generator = send_message(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")
    
