import streamlit as st
import requests

from langchain.document_loaders import YoutubeLoader
import os
import streamlit as st
import requests
import json
import time
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import random

BASE_URL = "http://localhost:8083"
Weaviate_endpoint = "/vector_DB_request/"

def process_text(text):
    # Remove quotes from the beginning and end of the text, if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Replace \n with an actual new line
    text = text.replace("\\n", "\n")

    return text

def display_user_classes(username, access_token):
    params = {
        "username": username,
        "mode": "display_classes",
        "vectorDB_type": "Weaviate",
        "mode": "display_classes",
        "class_name": "string"
        }
    file_path = None

    #headers = {"Authorization": f"Bearer {access_token}"}
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/",json=params, headers=headers)
        # Handle the response
    print("the response", resp, resp.content)

    response_content = resp.content.decode("utf-8")
    print("respcontent", response_content)
    user_classes = json.loads(response_content)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
        return user_classes
    else:
        print(resp.status_code, resp.content)
        return 
    
def send_vector_db_request(access_token, json_data, endpoint, uploaded_file=None):
    headers = {"Authorization": f"Bearer {access_token}"}


    response = requests.post(f"{BASE_URL}{endpoint}", data=json_data,headers=headers, files=uploaded_file)

    return response    

def authentication(username, password):
    data = {"username": username, "password": password}
    resp = requests.post(
        f"{BASE_URL}/token", data=data
    )
    if "access_token" not in resp.json():
        return None
    return resp.json()["access_token"]

def add_user(username, password,token_limit,access_token):

    headers = {"Authorization": f"Bearer {access_token}"}
    query_data = {
  "username": username,
  "password": password,
  "token_limit": token_limit
    }
    resp = requests.post(f"{BASE_URL}/db_request/add_user/", json=query_data, headers=headers)
    return True

def get_all_users_info(access_token):   
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(f"{BASE_URL}/db_request/get_all_users/", headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        return None

def retrieve_latest_conversation(username, access_token):
    query_data = {
  "username": username
}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/get_user_conversations/",json=query_data, headers=headers)
    if resp.status_code == 200:
        conversations = resp.json()['conversations']
        names = [d["name"] for d in conversations]
        if names:
            return {"names": names, "conversations": conversations}
        return None
    else:
        return None
    
def add_conversation(username, conversation,access_token):
    query_data = {
  "username": username,
  "content": json.dumps(conversation),
  "conversation_name": "Current Conversation",
}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/add_conversation/",json=query_data, headers=headers)
    if resp.status_code == 200:
        return True
    else:
        return False
def find_conversation_number(conversation, conversation_name):
    index = None
    for i, d in enumerate(conversation):
        if d["name"] == conversation_name:
            index = d["number"]
            break

    if index is not None:
        return index

def update_conversation_name(username, conversation_number, new_name, access_token):
    query_data = {
  "username": username,
  "conversation_number": conversation_number,
  "conversation_name": new_name
}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/update_conversation_name/",json=query_data, headers=headers)
    if resp.status_code == 200:
        return True
    else:
        return False

def retrieve_conversation(username, conversation_number, access_token):
    query_data = {
  "username": username,
  "conversation_number": conversation_number
}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/retrieve_conversation/",json=query_data, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        return None

def delete_conversation(username, conversation_number, access_token):
    query_data = {
  "username": username,
  "conversation_number": conversation_number
}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.delete(f"{BASE_URL}/db_request/delete_conversation/",json=query_data, headers=headers)
    if resp.status_code == 200:
        return True
    else:
        return False

if "username" not in st.session_state or st.sidebar.button("Logout"):
    # Login form
    if "username" not in st.session_state:
        username = st.text_input("Enter your username:")
        password = st.text_input('Enter your password', type='password') 
    else:
        username = st.session_state.username

    if  st.button("Login"):  # Add password field
                token =  authentication(username, password)
                if token:
                    st.session_state.token = token
                    st.session_state.username = username
                    st.session_state.show_logged_in_message = True
                else:
                    st.error("Invalid User")

else:
    if "show_logged_in_message" not in st.session_state:
        st.session_state.show_logged_in_message = False

    if st.session_state.show_logged_in_message:
        logedin_username = st.session_state.username
        classes = display_user_classes(st.session_state.username, st.session_state.token)
        print('classes', classes)

        if logedin_username == "admin":
            # Display the form to add new users
            new_user = st.text_input("Enter a new user:")
            new_user_password = st.text_input("Enter password for user:", type="password")
            token_limit = st.slider('Adjust the token limit', min_value=0, max_value=100000)

            if st.button("Add User") and new_user and new_user_password:
                # Add the new user to the list of valid users
                new_user = add_user(new_user,new_user_password,token_limit, st.session_state.token)
                if new_user:
                    st.success("New user added successfully!")
                else:
                    st.error("User already exists")
            # if st.button("Show Data"):
                # Get the tokens
                # tokens = get_user_tokens()
                # st.table(tokens)

        else:
            conversation_info = retrieve_latest_conversation(st.session_state.username, st.session_state.token)
            

            if conversation_info == None:
                history = StreamlitChatMessageHistory(key="langchain_messages")
                history.clear()
                history.add_ai_message("How can I help you?")
                ingest_to_db = messages_to_dict(history.messages)
                add_conversation(st.session_state.username, ingest_to_db, st.session_state.token)
            conversation_info = retrieve_latest_conversation(st.session_state.username,st.session_state.token)

                        # Display the buttons
            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            st.sidebar.subheader("User Access Token:")
            show_token = st.sidebar.button("Token")
            if show_token:
                st.sidebar.code(st.session_state.token)


            # Display the buttons
            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            st.sidebar.subheader("Previous Chats:")

#             # Add a new chat button
            if st.sidebar.button("New Chat"):
                latest_conversation_number = find_conversation_number(
                    conversation_info["conversations"], "Current Conversation"
                )

                update_conversation_name(
                    st.session_state.username,
                    latest_conversation_number,
                    f"Previous Conversation: {random.randint(0,10000)}",
                    st.session_state.token
                )
                history = StreamlitChatMessageHistory(key="langchain_messages")
                history.clear()
                history.add_ai_message("How can I help you?")
                ingest_to_db = messages_to_dict(history.messages)
                add_conversation(st.session_state.username, ingest_to_db, st.session_state.token)
                conversation_info = retrieve_latest_conversation(st.session_state.username,st.session_state.token)

            # Delete a chat
            if st.sidebar.button("Delete Chat"):
                delete_conversation(
                    st.session_state.username, st.session_state.conversation_number, st.session_state.token
                )
                conversation_info = retrieve_latest_conversation(st.session_state.username,st.session_state.token)
                if conversation_info == None:
                    history = StreamlitChatMessageHistory(key="langchain_messages")
                    history.clear()
                    history.add_ai_message("How can I help you?")
                    ingest_to_db = messages_to_dict(history.messages)
                    add_conversation(st.session_state.username, ingest_to_db)
                    conversation_info = retrieve_latest_conversation(st.session_state.username,st.session_state.token)

            selected_button_label = st.sidebar.radio(
                "Select a conversation:",
                conversation_info["names"],
                index=len(conversation_info["names"]) - 1,
            )

            # st.sidebar.markdown("<br>", unsafe_allow_html=True)

            if selected_button_label:
                st.session_state.newchat = "true"
                st.session_state.conversation_number = find_conversation_number(
                    conversation_info["conversations"], selected_button_label
                )
                st.empty()  # Clear the page content
                selected_value = retrieve_conversation(
                    st.session_state.username, st.session_state.conversation_number, st.session_state.token
                )["content"]

                history = StreamlitChatMessageHistory(key="langchain_messages")
                history.clear()
                if selected_value:
                    retrieve_from_db = json.loads(selected_value)
                    retrieved_messages = messages_from_dict(retrieve_from_db)
                    history.add_message(retrieved_messages)
                    st.text(history.messages)
                    for msg in history.messages:
                        len_message = len(msg)
                        
                        for agent in range(0, len_message-1): 
                            st.chat_message(msg[agent].type).write(msg[agent].content)

                    st.session_state.messages = [
                        {"role": "assistant", "content":history.messages[-1][-1].content}
                    ]

            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            llm_options = ["Llama_70b", "Llama_13b"]
            selected_llm = st.sidebar.selectbox("Choose a LLM :", llm_options, index=0)
            st.session_state.llm = selected_llm
            # st.image("Eschercloud.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            st.session_state.AI_Assistance = True
            search_choice = st.sidebar.radio(
                options=["AI Assistance", "Document Search"], label="Type of search"
            )
            if search_choice == "Document Search":
                classes = display_user_classes(st.session_state.username, st.session_state.token)
                print('classes', classes)
                selected_collection = st.sidebar.selectbox(
                    "select collection", classes['response']
                )
                if st.sidebar.button("Selected_class"):
                    if selected_collection is not None:
                        params = {
                            "username": st.session_state.username,
                            "collection_name": selected_collection,
                        }

                # collection_list = get_collections()
                # selected_collection = st.sidebar.selectbox(
                #     "select collection", collection_list
                # )
                st.session_state.AI_Assistance = False
            else:
                selected_collection = None
                st.session_state.AI_Assistance = True

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if prompt := st.chat_input():  # (disabled=not replicate_api):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        
                        username = st.session_state.username
                        conversation_number = st.session_state.conversation_number
                        newchat = st.session_state.newchat
                        params = {
                            "username": st.session_state.username,
                            "prompt": prompt,
                            "memory": st.session_state.newchat,
                            "conversation_number": st.session_state.conversation_number,
                            "AI_assistance": st.session_state.AI_Assistance,
                            "collection_name": selected_collection,
                            "llm_model": st.session_state.llm
                        }
                        URL = f"{BASE_URL}/llm_request"
                        headers = {"Authorization": f"Bearer {st.session_state.token}"}
                        response = requests.post(URL, json=params, headers=headers)
                        
                        response = response.content.decode()
                        if response:
                            response = json.loads(response)     
                            response = response["data"]
                            placeholder = st.empty()
                            full_response = process_text(response)
                            placeholder.markdown(full_response)

                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)
