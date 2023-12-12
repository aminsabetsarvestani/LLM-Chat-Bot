import subprocess
import json
import bibtexparser

import os
import PyPDF2
import arxiv
import time
from urllib.error import HTTPError

import arxiv
from fastapi import FastAPI
from tqdm import tqdm

from PyPDF2.errors import PdfStreamError
import pypdf
from langchain.document_loaders import PyPDFLoader
from PyPDF2.errors import PdfReadError, PdfStreamError

import weaviate
import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline

from langchain.chains import ChatVectorDBChain,RetrievalQA
import ray
import shutil
import os
import re
from weaviate.util import generate_uuid5
import time
from ray import serve
import os
import yaml
from typing import Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Security
import requests
import os
import yaml
import secrets
import zipfile
from typing import Optional
import pathlib
from sqlalchemy.orm import Session
import weaviate
import logging
import ray
import binascii
from typing import Any, List
import pypdf
import ray
from ray import serve
import os
import weaviate
from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter
import yaml
import time
from langchain.document_loaders import TextLoader
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from typing import Optional
from pydantic import BaseModel
from backend_database import Database
import zipfile
import os
from io import BytesIO
import shutil
import logging
from langchain.document_loaders import PyPDFLoader

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)

    
MAX_FILE_SIZE = config.max_file_size * 1024 * 1024  

class ArxivInput(BaseModel):
    username: str 
    class_name: Optional[str]
    query: Optional[str]
    paper_limit: Optional[int]
    recursive_mode: Optional[int] 
    mode: Optional[str]
    file_path: Optional[str] = None

Arxiv_app = FastAPI()

class WeaviateEmbedder:
    def __init__(self):
        self.time_taken = 0
        self.text_list = []
        self.weaviate_client = weaviate.Client(
            url="http://localhost:8080",   
        )

    def adding_weaviate_document(self, text_lst, collection_name):
        start_time = time.time()
        self.weaviate_client.batch.configure(batch_size=100)

        with self.weaviate_client.batch as batch:
            for text in text_lst:
                    batch.add_data_object(
                        text,
                        class_name=collection_name, 
                        uuid=generate_uuid5(text),
        )
        self.text_list.append(text)
        self.time_taken = time.time() - start_time
        return self.text_list

    def get(self):
        return self.lst_embeddings
    
    def get_time_taken(self):
        return self.time_taken

@ray.remote(num_gpus=0.1, num_cpus=12)
class WeaviateRayEmbedder:
    def __init__(self):
        self.time_taken = 0
        self.text_list = []
        # adding logger for debugging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        try:
            self.weaviate_client = weaviate.Client(
                url="http://localhost:8080",   
            )
        except:
            self.logger.error("Error in connecting to Weaviate")

    def adding_weaviate_document(self, text_lst, collection_name):
        self.weaviate_client.batch.configure(batch_size=100)

        with self.weaviate_client.batch as batch:
            for text in text_lst:
                batch.add_data_object(
                    text,
                    class_name=collection_name, 
                        #uuid=generate_uuid5(text),
        )
                self.text_list.append(text)
        #self.logger.info(f"Check the data that is being passed {self.text_list}: %s", )
        results= self.text_list
        ray.get(results)
        return self.text_list

    def get(self):
        return self.lst_embeddings
    
    def get_time_taken(self):
        return self.time_taken
@serve.deployment(ray_actor_options={"num_gpus":0.1}, autoscaling_config={
        #"min_replicas": config.VD_min_replicas,
        "initial_replicas": 1,
        #"max_replicas": config.VD_max_replicas,
        #"max_concurrent_queries": config.VD_max_concurrent_queries,
        })
@serve.ingress(Arxiv_app)
class ArxivSearch:
    def __init__(self):

        self.weaviate_client = weaviate.Client(
            url=config.weaviate_client_url,   
        )
        self.weaviate_vectorstore = Weaviate(self.weaviate_client, 'Chatbot', 'page_content', attributes=['page_content'])
        self.num_actors = config.Arxiv_base_actor_num
        self.chunk_size = config.VD_chunk_size
        self.chunk_overlap = config.VD_chunk_overlap
        self.database = Database()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
    
    def run_anystyle(self,input_pdf, base_dir):
        try:
            new_directory_path = os.path.join(base_dir, 'bib_files')
            os.makedirs(new_directory_path, exist_ok=True)
            
            command = ['anystyle', '-f', 'bib', 'find', input_pdf, new_directory_path]
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Check for successful execution
            if result.returncode == 0:
                output_file_name = os.path.basename(input_pdf).replace('.pdf', '.' + 'bib')
                output_file_path = os.path.join(new_directory_path, output_file_name)
                return output_file_path
            else:
                return f"Command failed with return code {result.returncode}."

        except subprocess.CalledProcessError as e:
            # Handle errors
            return f"An error occurred: {e.stderr}"

    def process_bib_files(self, file):
        try:
            with open(file) as bibtex_file:
                bib_database = bibtexparser.load(bibtex_file)

            references = []
            for entry in bib_database.entries:
                authors = self.clean_author_names(entry.get("author", ""))
                title = self.clean_title(entry.get("title", "No title"))
                references.append({"authors": authors, "title": title})

            return references
        
        except FileNotFoundError:
            return "BibTeX file not found."
        except Exception as e:
            return f"An error occurred: {e}"
        
    def clean_author_names(self, author_string):
        cleaned_authors = []
        for author in author_string.split(" and "):
            parts = [part for part in author.replace(',', '').split() if len(part.replace('.', '')) > 1 and not all(c.isupper() for c in part.replace('.', ''))]
            cleaned_authors.extend(parts)

        return cleaned_authors[:4]

    def clean_title(self, title_string):
        cleaned_title = [part.replace('title:', '').strip() for part in title_string.split(":")]
        return cleaned_title

    def is_close_match(self, result_title, query_title, result_author, query_author):
        return (query_title.lower() in result_title.lower()) and (query_author.lower() in result_author.lower())

    def arxiv_search(self, titles, authors, dir, count):
        for title in titles:
            for author in authors:
                try:
                    search_query = f"au:{author} AND ti:{title}"
                    search_results = arxiv.Search(query=search_query, max_results=1)

                    for result in tqdm(search_results.results()):
                        result_title = result.title
                        result_author = ', '.join([a.name for a in result.authors])
                        print(f"Title: {result_title}, Authors: {result_author}")
                        if self.is_close_match(result_title, title, result_author, author):
                                try:
                                    result.download_pdf(dirpath=dir)
                                    count += 1
                                    self.logger.info(f"count {count}: %s", )
                                    return count # Exit the loop once a match is found and downloaded
                                except FileNotFoundError:
                                    print("File not found.")
                                except HTTPError:
                                    print("Access forbidden.")
                                except ConnectionResetError:
                                    print("Connection reset by peer. Retrying in 5 seconds.")
                                    time.sleep(5)
                                    continue  # Retry the current iteration
                        continue #break  # Break loop if a search is completed

                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue  
            continue
        return count

    def weaviate_serialize_document(self, doc):
            document_title = doc.metadata.get('source', '').split('/')[-1]
            return {
                "page_content": doc.page_content,
                "document_title": document_title,

            }

    def weaviate_split_multiple_pdf(self, docs):    
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        text_docs = text_splitter.split_documents(docs)

        serialized_docs = [
                    self.weaviate_serialize_document(doc) 
                    for doc in text_docs
                    ]
        return serialized_docs	

    def split_document(self, docs, doc_name):        

            loader = PyPDFLoader(docs)

            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

            text_docs = text_splitter.split_documents(documents)
            serialized_docs = [
                self.weaviate_serialize_document(doc,doc_name) 
                for doc in text_docs
                ]
            return serialized_docs	

    def process_and_remove_pdfs(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory, filename)

                # Apply the split_document function
                doc_name = filename[:-4]  # Remove '.pdf' from filename to get the document name
                try:
                    serialized_docs = self.split_document(file_path, doc_name)
                    # Process serialized_docs as needed
                    print(f"Processed {filename}")

                    # Remove the PDF file after processing
                    os.remove(file_path)
                    print(f"Removed {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    def parse_pdf(self, dir):    
        documents = []
        for file in os.listdir(dir):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(dir, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                except pypdf.errors.PdfStreamError as e:
                    print(f"Skipping file {file} due to error: {e}")
                    continue  # Skip this file and continue with the next one
            elif file.endswith('.txt'):
                text_path = os.path.join(dir, file)
                try:
                    loader = TextLoader(text_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error in file {file}: {e}")
                    continue
        return documents

    def divide_workload(self, num_actors, documents):
        docs_per_actor = len(documents) // num_actors

        doc_parts = [documents[i * docs_per_actor: (i + 1) * docs_per_actor] for i in range(num_actors)]

        if len(documents) % num_actors:
            doc_parts[-1].extend(documents[num_actors * docs_per_actor:])

        return doc_parts

    def weaviate_embedding(self, text, cls):
        embedder = WeaviateEmbedder()
        embedder.adding_weaviate_document(text, cls)

    def weaviate_ray_embedding(self, text,cls):
        actor_workload = self.divide_workload(4, text)
        actors = [WeaviateRayEmbedder.remote() for _ in range(4)]
        [actor.adding_weaviate_document.remote(doc_part, cls) for actor, doc_part in zip(actors, actor_workload)]

    def merge_all_pdfs_into_final_dir(self, final_dir, base_dir):
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)

        # Regular expression to match iteration directories
        iter_dir_pattern = re.compile(r'^iteration_\d+$')

        # List all directories that match the iteration pattern with full paths
        all_iter_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, d)) and iter_dir_pattern.match(d)]
        for iter_dir in all_iter_dirs:
            for pdf_file in os.listdir(iter_dir):
                if pdf_file.endswith('.pdf'):
                    src_file_path = os.path.join(iter_dir, pdf_file)
                    dest_file_path = os.path.join(final_dir, pdf_file)

                    file_index = 1
                    base_name, extension = os.path.splitext(dest_file_path)
                    while os.path.exists(dest_file_path):
                        dest_file_path = f"{base_name}_{file_index}{extension}"
                        file_index += 1

                    shutil.move(src_file_path, dest_file_path)

    def query_arxiv_documents(self, query):
        self.doc_lst = []

        search = arxiv.Search(
        query = "quantum",
        max_results = 10,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )

        for result in arxiv.Client().results(search):
            self.doc_lst.append(result)
        return self.doc_lst

    def arxiv_pipeline(self, input_pdf, cls, ray=False, recursive=False, iteration = None, paper_limit = None):
        
            """Process all on one actor"""
            self.logger.info(f"the paper limit test {paper_limit}: %s")
            self.current_paper_count = 0
            self.logger.info(f"the paper limit test and paper count {self.current_paper_count}: %s")
            current_iter = 1
            base_dir = input_pdf
            for file in os.listdir(input_pdf):
                if file.endswith('.pdf'):
                    input_pdf_path = os.path.join(input_pdf, file)
                    break
                else:
                    return "No PDF file found in the directory."
            if not recursive:
                self.logger.info(f"testing checkpoint rec no %s")
                anystyle_output = self.run_anystyle(input_pdf_path,base_dir)
                parsed_data = self.process_bib_files(anystyle_output)
                for ref in parsed_data:
                    self.arxiv_search(ref['title'], ref['authors'])

                parsed_text = self.parse_pdf()
                serialized_text = self.weaviate_split_multiple_pdf(parsed_text)

                if ray == False:
                    self.weaviate_embedding(serialized_text, cls)

                elif ray is True:
                    self.weaviate_ray_embedding(serialized_text, cls)


                for filename in os.listdir('./pdfs/'):
                    file_path = os.path.join('./pdfs', filename)
                    if os.path.isfile(file_path) and filename.endswith(".pdf"):
                        os.remove(file_path)

            if recursive and iteration > 0:
                base_dir = input_pdf
                self.logger.info(f"the paper limit test {paper_limit}: %s")

                self.current_paper_count = 0
                while current_iter <= iteration:
                    self.logger.info(f"the paper limit test and paper count {self.current_paper_count}: %s")
                    if current_iter == 1:
                        iter_dir = os.path.join(input_pdf, f'iteration_{current_iter}')
                        if not os.path.exists(iter_dir):
                            os.makedirs(iter_dir)
                        anystyle_output = self.run_anystyle(input_pdf_path,base_dir)
                        parsed_data = self.process_bib_files(anystyle_output)
                        for ref in parsed_data:
                            if self.current_paper_count >= paper_limit:
                                break
                            self.current_paper_count = self.arxiv_search(ref['title'], ref['authors'], iter_dir, self.current_paper_count) or self.current_paper_count                           
                            self.logger.info(f"current_paper_count {self.current_paper_count}: %s", )

                        current_iter += 1
                    if self.current_paper_count >= paper_limit:
                                break   
                    elif current_iter >= 2:
                        iter_dir =  os.path.join(input_pdf, f'iteration_{current_iter}')
                        if not os.path.exists(iter_dir):
                            os.makedirs(iter_dir)
                    
                        previous_dir =  os.path.join(input_pdf, f'iteration_{current_iter - 1}')
                        pdf_files = [f for f in os.listdir(previous_dir) if f.endswith('.pdf')]
                        for pdf_file in pdf_files:
                            full_path = os.path.join(previous_dir, pdf_file)
                            anystyle_output = self.run_anystyle(full_path,base_dir)
                            
                            parsed_data = self.process_bib_files(anystyle_output)
                            for ref in parsed_data:
                                if self.current_paper_count >= paper_limit:
                                    break
                                if isinstance(ref, dict) and 'title' in ref and 'authors' in ref:
                                    self.current_paper_count = self.arxiv_search(ref['title'], ref['authors'], iter_dir, self.current_paper_count) or self.current_paper_count
                                    self.logger.info(f"current_paper_count 2.  {self.current_paper_count}: %s", )
                                else:
                                    print(f"Unexpected format of reference: {ref}")
                        current_iter += 1
                
                final_directory = input_pdf + '/final_pdfs'
                self.merge_all_pdfs_into_final_dir(final_directory, input_pdf)
                parsed_text = self.parse_pdf(final_directory)
                serialized_text = self.weaviate_split_multiple_pdf(parsed_text)
                if ray == False:
                    print('success split with no ray')
                # calling the weaviate embedder
                    self.weaviate_embedding(serialized_text, cls)

                elif ray is True:
                    print('splt with ray')
                    self.weaviate_ray_embedding(serialized_text, cls)

    @Arxiv_app.post("/")
    async def VectorDataBase(self, request: ArxivInput):
            try:
                if request.mode == "Search by query":
                    response  = self.query_arxiv_documents(request.query)
                    return response
                elif request.mode == "Upload file":
                    self.logger.info(f"request received {request}: %s", )
                    username = request.username
                    cls = request.class_name
                    full_class_name = f"{username}_{cls}"
                    response = self.arxiv_pipeline(request.file_path, full_class_name, ray=True, recursive=True, iteration=request.recursive_mode, paper_limit=request.paper_limit)
                    print('response', response)
                    return response
                self.logger.info(f"request processed successfully {request}: %s", )
                return {"username": request.username, "response": response}
            except Exception as e:
                self.logger.error("An error occurred: %s", str(e))