import json
import streamlit as st
import os
import requests
from langchain.document_loaders import YoutubeLoader
import logging
import pandas as pd
import time


logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

logger = logging.getLogger(__name__)
logger.propagate = True

port = 5001
BASE_URL = "http://localhost:8083"
Weaviate_endpoint = "/vector_DB_request/"
Arxiv_endpoint = "/arxiv_search/"

def replace_spaces_with_underscores(input_string):
    output_string = input_string.replace(" ", "_")
    return output_string

def add_class(username, class_name, access_token):
    #clean_class_name = replace_spaces_with_underscores(str(class_name))
    #print('clean class name', clean_class_name)
    params = {
        "username": username,
        "vectorDB_type": "Weaviate",
        "class_name": class_name,
        "mode": "create_collection"
        }
    #print('query data', params)
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/",json=params, headers=headers)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)

def delete_class(username, class_name, access_token):
    params = {
        "username": username,
        "class_name": class_name,
        "vectorDB_type": "Weaviate",
        "mode": "delete_class"
        }
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/",json=params, headers=headers)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)

def display_documents(username, class_name, access_token):
    params = {
        "username": username,
        "class_name": class_name,
        "vectorDB_type": "Weaviate",
        "mode": "display_documents"
        }
    

    #print("collection selected:", params)

    headers = {"Authorization": f"Bearer {access_token}"}
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/get_docs_in_class/",json=params, headers=headers)
    

    #print("the response", resp, resp.content)

    response_content = resp.content.decode("utf-8")
    #print("respcontent", response_content)
    document_list = json.loads(response_content)

    return document_list

def display_user_classes(username, access_token):
    params = {
        "username": username,
        "vectorDB_type": "Weaviate",
        "mode": "display_classes",
        "class_name": "string"
        }
    file_path = None

    #print("data:", params)

    #headers = {"Authorization": f"Bearer {access_token}"}
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/",json=params, headers=headers)
        # Handle the response
    #print("the response", resp, resp.content)

    response_content = resp.content.decode("utf-8")
    #print("respcontent", response_content)
    user_classes = json.loads(response_content)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
        return user_classes
    else:
        print(resp.status_code, resp.content)
        return 

def delete_document(username, class_name, document_name, access_token):
    params = {
        "username": username,
        "class_name": class_name,
        "mode": "delete_document",
        "vectorDB_type": "Weaviate",
        "file_title": document_name
        }
    print('params', params)
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)


def upload_documents(username, class_name, access_token, file_path):
    files = {'file': (file_path.name, file_path, file_path.type)}
    params = {
        "username": username,
        "class_name": class_name,
        "mode": "add_to_collection",
        "vectorDB_type": "Weaviate",
        }
    #print('file path', file_path)
    #print('file name', file_path.name)
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint,files)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/",json=params, headers=headers)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
        
    else:
        print(resp.status_code, resp.content)

def query_arxiv(access_token, arxiv_mode, query, username):
    params = {
        "username": username,
        "mode": arxiv_mode,
        "query": query,
        }
    resp = send_vector_db_request(access_token, params, Arxiv_endpoint) 
    response_content = resp.content.decode("utf-8")
    parsed_response = json.loads(response_content)
    all_entries = []  # List to store all entries

    # Check if there are entries in the response
    if 'response' in parsed_response and parsed_response['response']:
        # Iterate through each entry
        for entry in parsed_response['response']:
            # Extracting specific information from each entry
            title = entry.get('title', 'No Title')
            authors = [author['name'] for author in entry.get('authors', [])]
            summary = entry.get('summary', 'No Summary')
            url = entry.get('entry_id', 'No URL')

            # Create a dictionary for the current entry
            entry_dict = {
                "title": title,
                "authors": authors,
                "summary": summary,
                "url": url
            }

            # Add the dictionary to the list
            all_entries.append(entry_dict)
    else:
        print("No entries found in the response.")

    # Return the list of all entries
    return all_entries

def arxiv_search(username, class_name, access_token, arxiv_mode, arxiv_recusrive_mode, arxiv_paper_limit, query=None, file_path=None):
    if query is not None and file_path is None:
        print('it went in the 1st condition')
        params = {
            "username": username,
            "class_name": class_name,
            "mode": arxiv_mode,
            "recursive_mode": arxiv_recusrive_mode,
            "paper_limit": arxiv_paper_limit,
            "query": query,
            }
        resp = send_vector_db_request(access_token, params, Arxiv_endpoint)
    if file_path is not None and query is None: 
        print('it went in the 2nd condition')
        params = {
            "username": username,
            "class_name": class_name,
            "mode": arxiv_mode,
            "recursive_mode": arxiv_recusrive_mode,
            "paper_limit": arxiv_paper_limit,
            }
        resp = send_vector_db_request(access_token, params, Arxiv_endpoint, file_path)
    print('file path', file_path)
    
    
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)

def send_vector_db_request(access_token, json_data, endpoint, uploaded_file=None):
    headers = {"Authorization": f"Bearer {access_token}"}
    print('json data', json_data)

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

def display_colleciton_in_table():
    query_data = {
        "data_type": "Collection", 
        "mode": "get_all"
        }
    
    PORT = 8000
    data_type = "Collection"
    mode = "get_all"
    POST_URL = (
        f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&mode={mode}"
    )
    response = requests.post(POST_URL)

    collections_data = response.json()
    # selected_collections = st.multiselect("Select Collections:", response)
    collections_list = collections_data.get("collections", [])
    return collections_list

st.set_page_config(layout="wide")
st.title("Vector Database Management Dashboard")

# Fetch and display collections
st.sidebar.header("Collections")

def display_documents_in_col2(doc_list):
    if doc_list and doc_list['response']:
        with col2_2:
            st.table(doc_list['response'])

PORT = 8000

if "username" not in st.session_state or st.sidebar.button("Logout"):
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
                    #st.session_state.show_logged_in_message = True
                else:
                    st.error("Invalid User")
else:
        if "show_logged_in_message" not in st.session_state:
            st.session_state.show_logged_in_message = False

        if st.session_state.show_logged_in_message:
            logedin_username = st.session_state.username
        with st.expander("Manage collections", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                username = st.session_state.username
                print('user', username)
                st.header(f"{username} manageme")
                class_name = st.text_input("Enter collection name:")
                class_name = class_name.strip().replace(" ", "_")
                if st.button("Create Collection"):
                    if class_name is not None:
                        add_class(st.session_state.username, str(class_name), st.session_state.token)
                        st.success(f"Collection {class_name} created!")

                collection_to_delete = st.text_input("Collection to delete:")
                if st.button("Delete Collection"):
                    if collection_to_delete is not None:
                        delete_class(st.session_state.username, str(collection_to_delete), st.session_state.token)
                        st.success(f"Collection {collection_to_delete} deleted!")

            with col2:
                classes = display_user_classes(st.session_state.username, st.session_state.token)
                st.table(classes['response'])                

        with st.expander("Manage documents in collection", expanded=True):
            classes = display_user_classes(st.session_state.username, st.session_state.token)
            selected_collections = st.selectbox("Select Collections to manage:", classes['response'])
           
            col1_2, col2_2 = st.columns([1, 1.5])
            #selected_collections = st.text_input("Select Collections:")
            #if st.button("Display class"): 
            doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)   
            if selected_collections is not None:
                with col1_2:
                    st.session_state.doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)
                    st.subheader("Add Document")
                    uploaded_files = st.file_uploader("Add files", accept_multiple_files=True)
                    if uploaded_files:
                        st.session_state['uploaded_files'] = uploaded_files

                    if st.button("Upload documents"):
                        if uploaded_files:
                            for uploaded_file in st.session_state['uploaded_files']:
                                upload_documents(st.session_state.username, str(selected_collections), st.session_state.token, uploaded_file)
                                time.sleep(5)
                                st.text(f"Document {uploaded_file.name} Uploaded! ✅")
                            st.session_state.doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)

                    st.subheader("Remove Document")
                    if st.session_state.doc_list.get('response'):
                        st.session_state['selected_remove_file'] = st.selectbox("Select document:", st.session_state.doc_list['response'], key="remove_file_selectbox")

                    if st.button("Remove document"):
                        if st.session_state['selected_remove_file'] is not None:
                            delete_document(st.session_state.username, str(selected_collections), str(st.session_state['selected_remove_file']), st.session_state.token)
                            st.text(f"Document {st.session_state['selected_remove_file']} removed! ✅")
                            st.session_state.doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)

                # Call the function to display documents in the second column
                with col2_2:
                    display_documents_in_col2(st.session_state.get('doc_list', {}))        
            # if selected_collections is not None:
            #         with col1_2:
            #             #doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)
            #             st.session_state.doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)
            #             st.subheader("Add Document")
            #             uploaded_files = st.file_uploader("Add files", accept_multiple_files=True)
            #             if uploaded_files:
            #                 st.session_state['uploaded_files'] = uploaded_files
            #             if st.button("Upload documents"):
            #                 if uploaded_files:
            #                     for uploaded_file in st.session_state['uploaded_files']:
            #                         upload_documents(st.session_state.username, str(selected_collections), st.session_state.token, uploaded_file)
                                    
                                    
            #                         st.text(f"Document {uploaded_file.name} Uploaded! ✅")
            #                     st.session_state.doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)
                        
            #             display_documents_in_col2(st.session_state.get(doc_list, {}))
            #             st.subheader("Remove Document")
            #             if st.session_state.doc_list['response']:
            #                 st.session_state['selected_remove_file'] = st.selectbox("Select document:", st.session_state.doc_list['response'], key="remove_file_selectbox")

            #             if st.button("Remove document"):
            #                 if st.session_state['selected_remove_file'] is not None:
            #                     delete_document(st.session_state.username, str(selected_collections), str(st.session_state['selected_remove_file']), st.session_state.token)
            #                     st.text(f"Document {st.session_state['selected_remove_file']} removed! ✅")
               
            #         with col2_2:
            #             st.table(st.session_state.doc_list['response'])
                        







                    #     selected_remove_file = st.selectbox("Select document:", doc_list['response'])
                    #     if st.button("Remove document"):
                    #         if selected_remove_file is not None:
                    #             delete_document(st.session_state.username, str(selected_collections), str(selected_remove_file), st.session_state.token)
                    #             st.text(f"Document {selected_remove_file} removed! ✅")
                   
                    # with col2_2:
                    #    # doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)
                    #     st.table(doc_list['response'])
