import json
import streamlit as st
import os
import requests
from langchain.document_loaders import YoutubeLoader
import logging
import pandas as pd

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
#download_arxiv_paper(st.session_state.username, str(selected_collections), st.session_state.token, arxiv_mode, arxiv_recursive_mode, arxiv_paper_limit, selected_url
def download_arxiv_paper(username, class_name, access_token, arxiv_mode, arxiv_recursive_mode, arxiv_paper_limit, url):
    params = {
        "username": username,
        "class_name": class_name,
        "paper_limit": arxiv_paper_limit,
        "recursive_mode": arxiv_recursive_mode,
        "mode": arxiv_mode,
        "url": url,
        }
    print('params', params)
    resp = send_vector_db_request(access_token, params, Arxiv_endpoint)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)

def arxiv_search(username, class_name, access_token, arxiv_mode, arxiv_recusrive_mode, arxiv_paper_limit, query=None, file_path=None):
    if file_path is not None: 
        print('checkpoint 1')
        files = {'file': (file_path.name, file_path, file_path.type)}
        #print('file path', file_path, 'params', params)
        params = {
            "username": username,
            "class_name": class_name,
            "mode": arxiv_mode,
            "recursive_mode": arxiv_recusrive_mode,
            "paper_limit": arxiv_paper_limit,
            }

        print('file path', file_path, 'params', params)
        resp = send_vector_db_request(access_token, params, Arxiv_endpoint, files)
    print('file path', file_path, 'params', params)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
    else:
        print(resp.status_code, resp.content)

def display_user_classes(username, access_token):
    params = {
        "username": username,
        "vectorDB_type": "Weaviate",
        "mode": "display_classes",
        "class_name": "string"
        }
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    response_content = resp.content.decode("utf-8")
    user_classes = json.loads(response_content)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
        return user_classes
    else:
        print(resp.status_code, resp.content)
        return 

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


st.set_page_config(layout="wide")

st.title("Arxiv Search")

# Fetch and display collections
st.sidebar.header("Collections")

# Display documents in the main area
st.header("Documents")





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
                    st.session_state.show_logged_in_message = True
                else:
                    st.error("Invalid User")
else:
        if "show_logged_in_message" not in st.session_state:
            st.session_state.show_logged_in_message = False

        if st.session_state.show_logged_in_message:
            logedin_username = st.session_state.username

        if 'query_res' not in st.session_state:
            st.session_state.query_res = None

        st.subheader("Populate collection with Arxiv papers")
        with st.expander("Populate collection with Arxiv papers", expanded=True):
            col1, col2, col3, col4, col5 = st.columns([0.75, 0.75, 1, 1.1, 2])
            with col1:
                print('username', st.session_state.username)
                classes = display_user_classes(st.session_state.username, st.session_state.token)
                selected_collections = st.selectbox("Select Collections:", classes['response'])
            
            with col2: 
                arxiv_mode = st.radio(options=["Search by query", "Upload file"], label="Type upload")
            with col3:
                #arxv_recursive_mode = st.text_input("Number of recursive calls", value="1")
                arxiv_recursive_mode = st.slider(
                    'Select the number of iterations',
                    0, 7, 1)
                st.write('Iterations selected:', arxiv_recursive_mode)
                print('arxiv_recursive_mode', arxiv_recursive_mode)
            with col4:
                #arxiv_paper_limit = st.text_input("Number of papers to add", value="10")
                arxiv_paper_limit = st.slider(
                    'Select the maximum number of papers to add',
                    0, 50, 1)
                st.write('Iterations selected:', arxiv_paper_limit)
                print('arxiv_recursive_mode', arxiv_paper_limit)
            with col5:
                if arxiv_mode == "Search by query":
                    query = st.text_input("Enter query")
                    search_pressed = st.button("Search")

                    if search_pressed:
                        # Call your query function and store the result in session state
                        st.session_state.query_res = query_arxiv(st.session_state.token, arxiv_mode, query, st.session_state.username)

                    if st.session_state.query_res is not None:
                        # Only display the dataframe if query results are available
                        df = pd.DataFrame(st.session_state.query_res)
                        if not df.empty:
                            st.dataframe(df, column_config={"title": "Paper titles", "authors": "Authors", "summary": "Summary", "url": "Url's"}, hide_index=True)
                            title_to_url = pd.Series(df.url.values, index=df.title.values).to_dict()

                            # Display the selectbox with titles
                            selected_title = st.selectbox("Select a paper", df['title'])

                            # Use the selected title to get the corresponding URL
                            if selected_title:
                                selected_url = title_to_url[selected_title]
                                st.write(f"Selected paper URL: {selected_url}")
                                arxiv_mode = "Download paper"
                        

                elif arxiv_mode == "Upload file":
                    query = st.file_uploader("Upload file", type=["pdf"])

                #populate_pressed = st.button("Populate collection")
                if st.button("Populate collection"):
                    if arxiv_mode == "Download paper":
                        print('selected url', selected_url)
                        selected_file_path = download_arxiv_paper(st.session_state.username, str(selected_collections), st.session_state.token, arxiv_mode, arxiv_recursive_mode, arxiv_paper_limit, selected_url)
                        print('selected file path', selected_file_path)
                    # arxiv_search(st.session_state.username, str(selected_collections), st.session_state.token, arxiv_mode, arxiv_recursive_mode, arxiv_paper_limit, query=query)
                        st.text(f"Collection populated! ✅")
                    if arxiv_mode == "Upload file":
                        arxiv_search(st.session_state.username, str(selected_collections), st.session_state.token, arxiv_mode, arxiv_recursive_mode, arxiv_paper_limit, file_path=query)
                        st.text(f"Collection populated! ✅")
    #ef arxiv_search(username, class_name, access_token, arxiv_mode, arxiv_recusrive_mode, arxiv_paper_limit, query=None, file_path=None):
                        #def download_arxiv_paper(username, class_name, access_token, arxiv_mode, arxiv_recursive_mode, arxiv_paper_limit, title, url):