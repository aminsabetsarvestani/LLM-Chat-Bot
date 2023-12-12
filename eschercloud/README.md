
# EscherCloud API Documentation
The EscherCloud API enables you to interact with LLM-as-a-Service through an API. This document provides a quick guide to getting started with the service.

### Installation
Install the eschercloud package using pip:
```
pip install eschercloud
```

### Quick Start
To begin using the LLM service, you need to initialize it as shown below:

```python
from eschercloud import llms

# Initialize the LLM service
agent = llms.llmService(base_url="http://localhost:8081", access_token="XXXX")

```

```base_url``` is the URL to your LLM service endpoint.

```access_token``` is required to authenticate as a valid user.


##### Formulate your query in a dictionary format:

```python
query = {
    "prompt": "Explain Neural Networks",  # Prompt to be sent to the LLM
    "memory": False,  # Set to True if you need to have a conversation considering the chat history
    "conversation_number": 0,  # Specify which previous conversation to use as context
    "AI_assistance": True,  # If True, LLM uses its own knowledge to answer, if False, it retrieves information from a vector database
    "collection_name": "Youtube Videos"  # Specify the collection in the vector database to retrieve information from
    "llm_model" : "Llama_70b" # 
}

response = agent.query_inference(query)

```
## Methods
Below are the methods available to interact with the LLM service:

| Method                        | Description                                         | Access Level | Required Keys in Request Dictionary               |
|-------------------------------|-----------------------------------------------------|--------------|---------------------------------------------------|
| `query_inference`             | Send a prompt to a LLM for inference                  | User         | `prompt`(srt), `memory`(bool), `conversation_number` (int), `AI_assistance` (bool),  `collection_name` (str)|
| `add_user`                    | Adds a new user to the system.                      | Admin        | `username`, `password`, `token_limit` (Optional)  |
| `update_token_limit`          | Updates the token limit for a user.                 | Admin        | `username`, `token_limit`              |
| `get_all_data`                | Retrieves all data from the database.               | Admin        | N/A                                               |
| `delete_user`                 | Deletes a user and all related content.             | Admin        | `username`                                        |
| `check_user_existence`        | Checks if a user exists in the database.            | Admin        | `username`                                        |
| `disable_user`                | Disables a user account.                            | Admin        | `username`                                        |
| `add_conversation`            | Adds a conversation to the database.                | User         | `content`, `conversation_name`                    |
| `delete_conversation`         | Deletes a conversation from the database.           | User         | `conversation_number`                             |
| `retrieve_conversation`       | Retrieves a conversation from the database.         | User         | `conversation_number`                             |
| `retrieve_latest_conversation`| Retrieves the latest conversation.                  | User         | N/A                                               |
| `update_conversation`         | Updates a conversation in the database.             | User         | `conversation_number`, `content`                  |
| `update_conversation_name`    | Updates the name of a conversation in the database. | User         | `conversation_number`, `new_name`                 |
| `get_user_conversations`      | Retrieves all conversations for a user.             | User         | N/A                                               |
