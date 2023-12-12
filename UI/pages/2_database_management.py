import json
import streamlit as st
import os
import requests
from langchain.document_loaders import YoutubeLoader
import logging

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

def add_class(username, class_name, access_token):
    params = {
        "username": username,
        "vectorDB_type": "Weaviate",
        "class_name": class_name,
        "mode": "create_collection"
        }
    print('query data', params)
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

    print("data:", params)

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

def upload_documents(username, class_name, access_token, file_path):
    files = {'file': (file_path.name, file_path, file_path.type)}
    params = {
        "username": username,
        "class_name": class_name,
        "mode": "add_to_collection",
        "vectorDB_type": "Weaviate",
        }
    print('file path', file_path)
    print('file name', file_path.name)
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint,files)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/",json=params, headers=headers)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)

def query_arxiv(access_token, arxiv_mode, query):
    params = {
        "mode": arxiv_mode,
        "query": query,
        }
    resp = send_vector_db_request(access_token, params, Arxiv_endpoint) 
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)

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


st.title("Vector Database Management Dashboard")

# Fetch and display collections
st.sidebar.header("Collections")

# Display documents in the main area
st.header("Documents")

col1, col2 = st.columns(2)
PORT = 8000
#data_type = "Collection"
#mode = "get_all"
#POST_URL = f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&mode={mode}"
#response = requests.post(POST_URL)

#collections_data = response.json()

#collections_list = collections_data.get("collections", [])
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
                    st.session_state.show_logged_in_message = True
                else:
                    st.error("Invalid User")
else:
        if "show_logged_in_message" not in st.session_state:
            st.session_state.show_logged_in_message = False

        if st.session_state.show_logged_in_message:
            logedin_username = st.session_state.username

        with col1:
            username = st.session_state.username
            print('user', username)
            st.header("user:", st.session_state.username)
            class_name = st.text_input("Enter collection name:")
            class_name = class_name.strip().replace(" ", "-")
            if st.button("Create Collection"):
                if class_name is not None:
                    add_class(st.session_state.username, str(class_name), st.session_state.token)
                    st.success(f"Collection {class_name} created!")
                    #documents = display_colleciton_in_table()

            collection_to_delete = st.text_input("Collection to delete:")
            if st.button("Delete Collection"):
                if collection_to_delete is not None:
                    delete_class(st.session_state.username, str(collection_to_delete), st.session_state.token)
                    st.success(f"Collection {collection_to_delete} deleted!")
                    #documents = display_colleciton_in_table()
                   # POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={collection_to_delete}&mode={mode}"
                   # response = requests.post(POST_URL)
                   # st.success(f"Collection {collection_to_delete} deleted!")
                  #  documents = display_colleciton_in_table()


        with col2:
            classes = display_user_classes(st.session_state.username, st.session_state.token)
            st.table(classes)
            st.header("Select collections to use")
            selected_collections = st.text_input("Select Collections:")
            if st.button("Display class"):
                if selected_collections is not None:
                    doc_list = display_documents(st.session_state.username, str(selected_collections), st.session_state.token)
                    print("doc list", doc_list)
                    # selected_collections = st.multiselect("Select Collections:", collections_list)
                    #documents = display_colleciton_in_table()  # Fetch actual documents
                    st.table(doc_list)

            
            st.subheader("Add Document")
            uploaded_files = st.file_uploader("Add files", accept_multiple_files=True)
            if st.button("Upload documents"):
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        upload_documents(st.session_state.username, str(selected_collections), st.session_state.token, uploaded_file)
                        st.text(f"Document {uploaded_file.name} Uploaded! ✅")
            st.subheader("Populate collection with Arxiv papers")
            
            arxiv_mode = st.radio(options= ["Search by query", "Upload file"], label="Type upload")
            arxiv_recusrive_mode = st.text_input("Number of recursive calls", value=1)
            arxiv_paper_limit = st.text_input("Number of papers to add", value=10)
            if arxiv_mode == "Search by query":
                query = st.text_input("Enter query")
                if st.button("test button"):
                    query_res = query_arxiv(st.session_state.token, arxiv_mode, query)
                    if query_res is not None: 
                        selected_options = st.multiselect("Select one or more options", query_res)
                        st.write("You selected:", selected_options)
                print("query", query, type(query))
            elif arxiv_mode == "Upload file":
                query = st.file_uploader("Upload file", type=["pdf"])
                print("query", query, type(query))
            if st.button("Populate collection"):
                if arxiv_mode == "Search by query":
                    arxiv_search(st.session_state.username, str(selected_collections), st.session_state.token, arxiv_mode, arxiv_recusrive_mode, arxiv_paper_limit, query=query)
                    st.text(f"Collection populated! ✅")
                if arxiv_mode == "Upload file":
                    arxiv_search(st.session_state.username, str(selected_collections), st.session_state.token, arxiv_mode, arxiv_recusrive_mode, arxiv_paper_limit, file_path=query)
                    st.text(f"Collection populated! ✅")
                

                #def arxiv_search(username, class_name, access_token, arxiv_mode, arxiv_recusrive_mode, arxiv_paper_limit, query=None, file_path=None):
         # uploaded_pdf = st.file_uploader("Add PDF", type=["pdf"])
          #  if st.button("Upload document"):
           #         if uploaded_pdf is not None:
            #            upload_documents(st.session_state.username, str(selected_collections), st.session_state.token, uploaded_pdf)
             #           st.text("Document Uploaded!" + " ✅")
        #st.header("Select collections to use")
        #doc_list = display_documents(selected_collections)
        #selected_collections = st.selectbox("Select Collections:", documents)

      #  doc_list = display_documents(selected_collections)

        # Get the documents in collection:

       # st.header("Collection selected", selected_collections)

'''''
        col1_2, col2_2 = st.columns(2)

        st.markdown("<br>", unsafe_allow_html=True)

        with col1_2:
            management_mode = st.radio(
                options=["Upload Document", "Upload Video", "Upload Webpage"],
                label="Type upload",
            )
            if management_mode == "Upload Document":
                st.subheader("Add Document")
                uploaded_pdf = st.file_uploader("Add PDF", type=["pdf"])
                document_name = st.text_input("Enter document name:")
                document_metadata = st.text_input("Enter document metadata:")
                if st.button("Upload document"):
                    if uploaded_pdf is not None:
                        pdf_filename = os.path.join(pdf_directory, uploaded_pdf.name)
                        # Save the uploaded PDF to the specified directory
                        with open(pdf_filename, "wb") as f:
                            f.write(uploaded_pdf.read())
                        PORT_NUMBER = 8000
                        mode = "add"
                        data_type = "Document"
                        collection_ = selected_collections
                        doc_name = document_name
                        POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={pdf_filename}&mode={mode}&collection={collection_}&doc_name={doc_name}"
                        response = requests.post(POST_URL)
                        st.text("Document Uploaded!" + " ✅")

                        doc_list = display_documents(selected_collections)
            elif management_mode == "Upload Video":
                st.subheader("Upload Videos")
                st.write("Enter YouTube video URLs below:")

                video_name = st.text_input("name of the video")
                user_input = st.text_input("Video URL:")

                if st.button("Upload Videos"):
                    upload_videos(user_input, port, selected_collections, video_name)

            elif management_mode == "Upload Webpage":
                st.subheader("Upload Webpage")
                st.write("Enter a webpage URLs below:")

                page_name = st.text_input("name of the page")
                url = st.text_input("page URL:")

                if st.button("Upload Webpage"):
                    upload_webpage(url, port, selected_collections, page_name)

            st.subheader("Remove Document")
            selected_document = st.selectbox("Select document:", doc_list)
            print("selected doc", selected_document)
            if st.button("Remove document"):
                if selected_document is not None:
                    PORT_NUMBER = 8000
                    mode = "remove"
                    data_type = "Document"
                    collection_rm = selected_collections
                    doc_rm_name = selected_document
                    print(
                        "collection name remove:",
                        collection_rm,
                        "doc name remove:",
                        doc_rm_name,
                        type(collection_rm),
                        type(doc_rm_name),
                    )
                    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&mode={mode}&remove={collection_rm}&doc={doc_rm_name}"
                    response = requests.post(POST_URL)
                    print("Da resp", response, response.content)
                    st.text("Document Removed!" + " ✅")

        with col2_2:
            doc_list = display_documents(selected_collections)
            st.header("Available documents in the collection")
            if doc_list:
                st.table(doc_list)
            else:
                st.write("No documents to display")

'''
