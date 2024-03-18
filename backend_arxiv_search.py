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
from langchain_community.document_loaders import TextLoader
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)

    
MAX_FILE_SIZE = config.max_file_size * 1024 * 1024  

class ArxivInput(BaseModel):
    username: Optional[str]
    class_name: Optional[str] = None
    query: Optional[str] = None
    paper_limit: Optional[int] = None
    recursive_mode: Optional[int] = None
    mode: Optional[str]
    title: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    dir_name: Optional[str] = None

Arxiv_app = FastAPI()

@ray.remote(num_cpus=0.07, num_gpus=0.008)
class Arxiv_actors:
    def __init__(self):
        self.time_taken = 0 
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

    async def run_ref(self, doc_list, base_dir):
        bib_files = await self.run_anystyle(doc_list, base_dir)
        #self.logger.info('made the bib files check dir', base_dir)
        references = await self.process_bib_files(bib_files)
        return references

    async def run_arxiv(self, references, base_dir, status_actor):
        count = 0
        for refs in references:
            if await status_actor.should_terminate.remote():
                self.logger.info("Termination signal received, stopping.")
                break
            count = await self.arxiv_search(refs['title'], refs['authors'], base_dir, count, status_actor)
        return count

    async def terminate_actors(self):
        ray.actor.exit_actor()

    async def run_anystyle(self, doc_list, base_dir):
        print('base directory', base_dir)
        output_paths = []

        try:
            new_directory_path = os.path.join(base_dir, 'bib_files')
            os.makedirs(new_directory_path, exist_ok=True)
            for input_file in doc_list:
                output_file_name = os.path.basename(input_file).replace('.pdf', '.bib')
                output_file_path = os.path.join(new_directory_path, output_file_name)
                command = ['anystyle', '-f', 'bib', 'find', input_file, new_directory_path]
                
                # asynchronously
                proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    output_paths.append(output_file_path)
                else:
                    output_paths.append(f"Command failed for {input_file} with return code {proc.returncode} and error {stderr.decode()}.")

        except subprocess.CalledProcessError as e:
            return [f"An error occurred: {e.stderr}"]

        return output_paths
    
    def clean_author_names(self, author_string):
        cleaned_authors = []
        for author in author_string.split(" and "):
            parts = [part for part in author.replace(',', '').split() if len(part.replace('.', '')) > 1 and not all(c.isupper() for c in part.replace('.', ''))]
            cleaned_authors.extend(parts)

        return cleaned_authors[:4]

    def clean_title(self, title_string):
            cleaned_title = [part.replace('title:', '').strip() for part in title_string.split(":")]
            return cleaned_title

    def divide_workload(self, num_actors, documents):
        docs_per_actor = len(documents) // num_actors

        extra_docs = len(documents) % num_actors

        doc_parts = []
        start_index = 0

        for i in range(num_actors):
            end_index = start_index + docs_per_actor + (1 if i < extra_docs else 0)
            doc_parts.append(documents[start_index:end_index])
            start_index = end_index

        return doc_parts

    def split_references(refs, num_actors):
        chunk_size = len(refs) // num_actors

        split_refs = []
        start = 0

        for i in range(num_actors):
            
            if i == num_actors - 1:
                end = len(refs)
            else:
                end = start + chunk_size

            split_refs.append(refs[start:end])
            
            start = end

        return split_refs

    async def process_bib_files(self, lst):
        references = []
        for file in lst:
            try:
                with open(file) as bibtex_file:
                    bib_database = bibtexparser.load(bibtex_file)

                for entry in bib_database.entries:
                    authors = self.clean_author_names(entry.get("author", ""))
                    title = self.clean_title(entry.get("title", "No title"))
                    references.append({"authors": authors, "title": title})

            except FileNotFoundError:
                print(f"Warning: BibTeX file not found and will be skipped: {file}")
                continue  
            except Exception as e:
                print(f"An error occurred while processing {file}: {e}")
                continue  
        print('refs', references)
        return references

    async def arxiv_pipeline(self, lst, base_dir,status_actor):
        current_paper_count = 0
        for refs in lst:
            if await status_actor.should_terminate.remote():
                print("Terminating actors")
                break
            if isinstance(refs, dict) and 'title' in refs and 'authors' in refs:
                current_paper_count = self.arxiv_search(refs['title'], refs['authors'], base_dir, 0, status_actor)
                print(f"Found {current_paper_count} papers")
            else:
                print(f"Unexpected format of refserence: {refs}")
        return current_paper_count

    def is_close_match(self, result_title, query_title, result_author, query_author):
        return (query_title.lower() in result_title.lower()) and (query_author.lower() in result_author.lower())

    def sync_arxiv_search(self, title, author, dir, count):
        client = arxiv.Client()
        search_query = f"au:{author} AND ti:{title}"
        search_results = arxiv.Search(query=search_query, max_results=1)
        for result in client.results(search_results):
            result_title = result.title
            result_author = ', '.join([a.name for a in result.authors])
            print(f"Title: {result_title}, Authors: {result_author}")
            if self.is_close_match(result_title, title, result_author, author):
                try:
                    result.download_pdf(dirpath=dir)
                    count += 1
                    return count  
                except Exception as e:
                    print(f"An error occurred during download: {e}")
        return count

    async def arxiv_search(self, titles, authors, dir, count, status_actor):
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            for title in titles:
                for author in authors:
                    # Check termination condition
                    if await status_actor.should_terminate.remote():
                        print("Terminating actors")
                        return count

                    future = loop.run_in_executor(executor, self.sync_arxiv_search, title, author, dir, count)
                    count = await future
        return count
    
    def get_function(self, res):
        ref = ray.get(res)
        return ref
    
    async def terminate(self):
        ray.actor.exit_actor()

@ray.remote
class SharedStatus:
    def __init__(self):
        self.terminate = False

    def set_terminate(self, value):
        self.terminate = value

    def should_terminate(self):
        return self.terminate
    
@ray.remote(num_cpus=0.05)
class DocumentCounter:
    def __init__(self, base_dir, max_docs=None):
        self.max_docs = max_docs
        self.base_dir = base_dir
        self.running = True
        self.total_docs = 0
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

    def start_counting(self, status_actor):
        while self.running:
            subdirs = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith('iteration_')]

            total_count = 0
            for subdir in subdirs:
                subdir_path = os.path.join(self.base_dir, subdir)
                doc_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                total_count += doc_count

            self.total_docs = total_count
            print(f"Total number of documents in subdirectories: {self.total_docs}")

            if self.max_docs and self.total_docs >= self.max_docs:
                ray.get(status_actor.set_terminate.remote(True))
                self.running = False
            self.logger.info(f"Total number of documents in subdirectories: {self.total_docs}")
            time.sleep(1)

    def get_document_count(self):
        print('total docs', self.total_docs)
        return self.total_docs

    def stop_counting(self):
        print('stopping')
        self.running = False

    def is_limit_reached(self):
        return self.total_docs >= self.max_docs if self.max_docs else False
    
class SharedProcessedDocs:
    def __init__(self):
        self.processed_docs = set()
        self.lock = threading.Lock()

    def add_processed_doc(self, document_path):
        with self.lock:
            self.processed_docs.add(document_path)
            print(f"Added to {self.add_processed_doc}")

    def has_been_processed(self, document_path):
        with self.lock:
            return document_path in self.processed_docs
        
@ray.remote(num_gpus=0.1)
class WeaviateRayEmbedder:
    def __init__(self, source_dir, destination_dir, check_interval, class_name=None):
        print('init 1')
        self.source_dir = source_dir
        self.check_interval = check_interval
        self.destination_dir = destination_dir
        self.time_taken = 0
        self.text_list = []
        self.processed_docs = set()
        self.running = True
        self.shared_processed_docs = SharedProcessedDocs()
        self.class_name = class_name
        #self.logger('')
        self.weaviate_client = weaviate.Client(
            url="http://localhost:8080",   
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        self.logger.info('finished init embedders')

    async def convert_file_to_text(self, document_path):
        documents = await self.parse_pdf(document_path)
        print('documents', documents)
        return documents
        

    async def run_embedder_on_text(self, documents):
        serialized_docs = await self.weaviate_split_pdf(documents)
        doc_list = await self.adding_weaviate_document(serialized_docs, self.class_name)
        return doc_list

    async def run_embedder(self):
        while self.running:
            document_path = await self.get_document_to_process()
            self.logger.info('checking document path', document_path)
            if document_path is not None:
                self.logger.info('checking weaviate embedder 2')
                if not self.shared_processed_docs.has_been_processed(document_path):
                    self.shared_processed_docs.add_processed_doc(document_path)
                    self.logger.info("processed files: ", self.shared_processed_docs.processed_docs)
                    document = await self.convert_to_text(document_path)
                    serialized_doc = await self.weaviate_split_pdf(document)
                    await self.adding_weaviate_document(serialized_doc, self.class_name)
            else:
                await asyncio.sleep(1)

    async def get_document_to_process(self):
        all_files = [os.path.join(self.source_dir, filename) for filename in os.listdir(self.source_dir)]

        unprocessed_files = [file for file in all_files if not self.shared_processed_docs.has_been_processed(file)]
        print('check the get doc method')
        if unprocessed_files:
            return unprocessed_files[0]
        else:
            return None        

    async def convert_to_text(self, document_path):   
        documents = []
        if document_path.endswith('.pdf'):
            try:
                loader = PyPDFLoader(document_path)
                documents.extend(loader.load())
                print('documents', documents)
            except pypdf.errors.PdfStreamError as e:
                print(f"Skipping file {document_path} due to error: {e}")
        return documents

    async def weaviate_split_pdf(self, docs):    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        text_docs = text_splitter.split_documents(docs)

        serialized_docs = [
                    await self.weaviate_serialize_document(doc) 
                    for doc in text_docs
                    ]
        return serialized_docs
    
    async def adding_weaviate_document(self, text_lst, collection_name):
        self.weaviate_client.batch.configure(batch_size=100)

        with self.weaviate_client.batch as batch:
            for text in text_lst:
                batch.add_data_object(
                    text,
                    class_name=collection_name, 
                        #uuid=generate_uuid5(text),
        )
                self.text_list.append(text)

        return self.text_list

    async def check_if_processed(self, doc):
        return doc in self.processed_docs

    async def mark_as_processed(self, document_path):
        self.processed_docs.add(document_path)

    async def terminate_actors(self):
        ray.actor.exit_actor()

    async def run_move_files(self):
        await self.move_pdfs()

    async def move_pdfs(self):
        while self.running:
            new_pdfs = await self.find_new_pdfs()
            
            if new_pdfs:
                await self.move_files(new_pdfs)

            await asyncio.sleep(self.check_interval)

    async def find_new_pdfs(self):
        new_pdfs = []
        processed_files = set() 

        for root, dirs, files in os.walk(self.source_dir):
            if root == self.source_dir or root.startswith(os.path.join(self.source_dir, "iteration_")):
                for file in files:
                    if file.endswith(".pdf"):
                        pdf_path = os.path.join(root, file)
                        if pdf_path not in processed_files:
                            new_pdfs.append(pdf_path)
                            processed_files.add(pdf_path) 

        return new_pdfs

    async def move_files(self, files):
        for file in files:
            destination = os.path.join(self.destination_dir, os.path.basename(file))
            shutil.copy(file, destination)

    async def stop(self):
        self.running = False

    async def weaviate_serialize_document(self, doc):
            document_title = doc.metadata.get('source', '').split('/')[-1]
            return {
                "page_content": doc.page_content,
                "document_title": document_title,
            }
    
    async def parse_pdf(self, file_path_list):    
        documents = []
        for pdf_path in file_path_list:
            #if pdf_path.endswith('.pdf'):
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                    self.logger.info('weaviate embedder doc length', len(documents))
                except pypdf.errors.PdfStreamError as e:
                    print(f"Skipping file {pdf_path} due to error: {e}")
                    continue  # Skip this file and continue with the next one
        return documents
    
    def get_time_taken(self):
        return self.time_taken
    
@serve.deployment(#ray_actor_options={"num_gpus":0.1}, autoscaling_config={
        #"min_replicas": config.VD_min_replicas,
        #"initial_replicas": 1,
        #"max_replicas": config.VD_max_replicas,
        #"max_concurrent_queries": config.VD_max_concurrent_queries,
#        }
)
@serve.ingress(Arxiv_app)
class ArxivSearch:
    def __init__(self):

        self.weaviate_client = weaviate.Client(
            url=config.weaviate_client_url,   
        )
        self.weaviate_vectorstore = Weaviate(self.weaviate_client, 'Chatbot', 'page_content', attributes=['page_content'])
        self.num_actors = 2
        self.chunk_size = config.VD_chunk_size
        self.chunk_overlap = config.VD_chunk_overlap
        self.actual_count = 0 
        self.full_path = None
        self.database = Database()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

    def get_pdf_paths(self,dir):
        pdf_paths = []
        for file in os.listdir(dir):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(dir, file)
                pdf_paths.append(pdf_path)
        return pdf_paths

    def split_workload(self,file_paths, num_actors):
        return [file_paths[i::num_actors] for i in range(num_actors)]

    def count_pdf_files(self,directory):
            try:
                self.logger.info(f'count pdf files, and check the directory {directory}')
                
                self.full_path = os.path.join(directory, '/')
                self.logger.info('full path', self.full_path)


                pdf_files = [os.path.join(self.full_path, f) for f in os.listdir(self.full_path) if f.endswith('.pdf')]
                self.logger.info('pdf files', pdf_files)

                actual_count = 0
                self.logger.info('actual count', actual_count)
                pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

                #print('count pdf', len(pdf_files))
                for filename in os.listdir(directory):
                    if filename.endswith(".pdf"):
                        self.logger.info('pdf files', filename)
                        actual_count += 1
                if actual_count % 2 != 0:
                    if actual_count > 1:    
                        count = (actual_count - 1) / 2
                        self.logger.info('count', count)
                    else:
                        count = actual_count  
                        self.logger.info('count', count)
                else:
                    count = actual_count / 2

                return actual_count, int(count), pdf_files
            except Exception as e:
                self.logger.error(f'Error in count_pdf_files: {e}')

    def count_bib_files(self,directory):
            count = 0
            for filename in os.listdir(directory):
                if filename.endswith(".bib"):
                    count += 1
            return count

    def divide_workload(self,num_actors, documents):
        docs_per_actor = len(documents) // num_actors

        extra_docs = len(documents) % num_actors

        doc_parts = []
        start_index = 0

        for i in range(num_actors):
            end_index = start_index + docs_per_actor + (1 if i < extra_docs else 0)
            doc_parts.append(documents[start_index:end_index])
            start_index = end_index

        return doc_parts

    def flatten_list_of_lists(self, list_of_lists):
        """Flatten a list of lists into a single list."""
        return [item for sublist in list_of_lists for item in sublist]

    def divide_into_equal_parts(self, lst, n):
        """Divide a list into n parts of equal length."""
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def query_arxiv_documents(self, query):
        self.doc_lst = []

        search = arxiv.Search(
            query = str(query),
            max_results = 3,
            sort_by = arxiv.SortCriterion.Relevance,
            sort_order = arxiv.SortOrder.Descending
        )

        for result in arxiv.Client().results(search):
            self.doc_lst.append(result)
        return self.doc_lst

    def download_arxiv_paper(self, url, dir_name):
        arxiv_id = url.split('/')[-1]
        search_result = arxiv.Search(id_list=[arxiv_id], max_results=1)
        results = list(search_result.results())    
        for result in tqdm(results):
            result.download_pdf(dirpath=dir_name)
            return dir_name  
        return None

    def arxiv_pipeline(self, input_pdf, cls, recursive=False, iteration = None, num_actors=None, paper_limit = None):
            import ray
            # init the pipeline

            current_iter = 1
            self.logger.info('checkpoint 1')
            base_dir = input_pdf
            self.logger.info('checkpoint 2')
            destination_dir = os.path.join(input_pdf, 'pdf_dir')
            if not os.path.exists(destination_dir):
                                os.makedirs(destination_dir) 
            self.logger.info('checkpoint 3') 
            self.logger.info(base_dir)
            self.logger.info('checkpoint 4')
            #shared_processed_docs = SharedProcessedDocs()
            self.logger.info('checkpoint 5')
            status_actor = SharedStatus.remote()
            self.logger.info('checkpoint 6')
            #weaviate_actor = WeaviateRayEmbedder.remote(base_dir, destination_dir, 1, None)
            self.logger.info('checkpoint 7')
            weaviate_embedders = [WeaviateRayEmbedder.remote(base_dir, destination_dir, 1, cls) for _ in range(int(3))]
            self.logger.info('checkpoint 8')
            count_actor = DocumentCounter.remote(base_dir, paper_limit)
            self.logger.info('checkpoint 9')
            count_actor.start_counting.remote(status_actor)
            self.logger.info('checkpoint 10')
            #weaviate_actor.run_move_files.remote()
            #[weaviate_embedder.run_embedder.remote() for weaviate_embedder in weaviate_embedders]
            self.logger.info(f'check if recursive {recursive}, and for iteration {iteration}')
            if recursive and iteration > 0:
                self.logger.info('checkpoint 11')
                while current_iter <= iteration:
                    self.logger.info('checkpoint 12')
                    if current_iter == 1:
                        self.logger.info('checkpoint 13')
                        a_cnt, cnt, lst = self.count_pdf_files(base_dir)
                        self.logger.info('checkpoint 14')
                        self.logger.info('count pdf', a_cnt, cnt, lst)  
                        iter_dir =  os.path.join(input_pdf, f'iteration_{current_iter}')
                        self.logger.info('checkpoint 15')
                        if not os.path.exists(iter_dir):
                                os.makedirs(iter_dir)  
                        self.logger.info('checkpoint 16')        
                        if a_cnt != 0:
                            self.logger.info('checkpoint 17')
                            #count_actor.start_counting.remote(status_actor)

                            parts = self.divide_workload(int(num_actors), lst)
                            #self.logger.info('parts', len(parts), parts)
                            actors = [Arxiv_actors.remote() for _ in range(int(num_actors))]
                            self.logger.info('actors', len(actors), actors)
                            workloads = [parts[i] for i in range(len(actors))]

                            futures = [actor.run_ref.remote(workload, base_dir) for actor, workload in zip(actors, workloads)]

                            ref = self.flatten_list_of_lists(ray.get(futures))
                            ref_div = self.divide_into_equal_parts(ref, len(actors))
                            
                            futures_1 = [actor.run_arxiv.remote(workload, iter_dir, status_actor) for actor, workload in zip(actors, ref_div)]

                            final_result_iter_1 = ray.get(futures_1)
                            [actor.terminate_actors.remote() for actor in actors]
                            self.logger.info(f'check the arxiv actors list {actors}')
                            

                            current_iter += 1
                            self.logger.info('finished 1st iter, current iter', current_iter)
                            if iteration == 1:
                                path_lst = self.get_pdf_paths(iter_dir)
                                path_lst = self.split_workload(path_lst, len(weaviate_embedders))
                                futures = [weaviate_embedder.convert_file_to_text.remote(i) for weaviate_embedder, i in zip(weaviate_embedders, path_lst)]
                                doc_lst =[weaviate_embedder.run_embedder_on_text.remote(workload) for weaviate_embedder, workload in zip(weaviate_embedders, futures)]
                                final_res_embedder = ray.get(doc_lst)
                                self.logger.info('doing embeddings 1')
                                [weaviate_embedder.terminate_actors.remote() for weaviate_embedder in weaviate_embedders]
                                #await run_embedding_pipeline(None, iter_dir, weaviate_embedders)
                                break
                            else: 
                                continue
                            
                    elif current_iter >= 2:
                        

                        iter_dir =  os.path.join(input_pdf, f'iteration_{current_iter}')
                        self.logger.info('iter_dir', iter_dir)
                        
                        if not os.path.exists(iter_dir):
                            os.makedirs(iter_dir)
                        
                        previous_dir =  os.path.join(input_pdf, f'iteration_{current_iter - 1}')

                             
                        self.logger.info('previous_dir', previous_dir)
                        a_cnt, cnt, lst = self.count_pdf_files(previous_dir)
                        self.logger.info('count pdf', a_cnt, cnt, lst)            
                        if a_cnt != 0:

                            self.logger.info('check the add dir')
                            #count_actor.add_directory.remote(iter_dir)
                            #time.sleep(5)
                            path_lst = self.get_pdf_paths(previous_dir)
                            path_lst =self.split_workload(path_lst, len(weaviate_embedders))
                            futures = [weaviate_embedder.convert_file_to_text.remote(i) for weaviate_embedder, i in zip(weaviate_embedders, path_lst)]
                            doc_lst = [weaviate_actor.run_embedder_on_text.remote(workload) for weaviate_actor, workload in zip(weaviate_embedders, futures)]
                            final_res_embedder = ray.get(doc_lst)

                            #await run_embedding_pipeline(None, previous_dir, weaviate_embedders)  
                            self.logger.info('doing embeddings 2')
                            self.logger.info(f'checking the actual count of a_cnt {a_cnt}')
                            parts = self.divide_workload(int(a_cnt), lst) #modified here
                            print('parts', len(parts), parts)
                            actors = [Arxiv_actors.remote() for _ in range(int(a_cnt))]
                            self.logger.info('actors', len(actors), actors)
                            workloads = [parts[i] for i in range(len(actors))]

                            futures = [actor.run_ref.remote(workload, base_dir) for actor, workload in zip(actors, workloads)]

                            ref = self.flatten_list_of_lists(ray.get(futures))
                            ref_div = self.divide_into_equal_parts(ref, len(actors))
                            
                            futures_2 = [actor.run_arxiv.remote(workload, iter_dir, status_actor) for actor, workload in zip(actors, ref_div)]
                            final_result_iter_1 = ray.get(futures_2)
                            self.logger.info('finished loop, current iter', current_iter)
                            if current_iter == iteration:
                                self.logger.info('doing embeddings 3')
                                #await run_embedding_pipeline(None, iter_dir, weaviate_embedders)  
                                #break
                                path_lst = self.get_pdf_paths(iter_dir)
                                path_lst = self.split_workload(path_lst, len(weaviate_embedders))
                                futures = [weaviate_embedder.convert_file_to_text.remote(i) for weaviate_embedder, i in zip(weaviate_embedders, path_lst)]
                                doc_lst = [weaviate_actor.run_embedder_on_text.remote(workload) for weaviate_actor, workload in zip(weaviate_embedders, futures)]
                                final_res_embedder = ray.get(doc_lst)
                                [weaviate_embedder.terminate_actors.remote() for weaviate_embedder in weaviate_embedders]
                                break
                            else:
                                current_iter += 1
                                continue

            if not recursive or iteration == 0:
                   self.logger.info('add logic to add a single file to Weaviate.')

    @Arxiv_app.post("/")
    async def VectorDataBase(self, request: ArxivInput):
            try:
                if request.mode == "Search by query":
                    response  = self.query_arxiv_documents(request.query)
                    return response
                elif request.mode == "Download paper":
                    self.logger.info(f"request received {request}: %s", )
                    paper_path = self.download_arxiv_paper(request.url, request.dir_name)
                    username = request.username
                    cls = request.class_name
                    full_class_name = f"{username}_{cls}"
                    self.logger.info(f'the paper path {paper_path}: %s',)
                    #arxiv_pipeline_1(None, "../Ray_demo/llmAPI/received_files/19a57ad7f08b371a/", "Papers_23", True, 2, 7, 40)
                    response = self.arxiv_pipeline(paper_path, full_class_name, recursive=True, iteration=request.recursive_mode, num_actors=5, paper_limit=request.paper_limit)
                    return response
                elif request.mode == "Upload file":
                    self.logger.info(f"request received {request}: %s", )
                    username = request.username
                    cls = request.class_name
                    full_class_name = f"{username}_{cls}"
                    response = self.arxiv_pipeline(request.file_path, full_class_name, recursive=True, iteration=request.recursive_mode, num_actors=5, paper_limit=request.paper_limit)
                    print('response', response)
                    return response
                self.logger.info(f"request processed successfully {request}: %s", )
                return {"username": request.username, "response": response}
            except Exception as e:
                self.logger.error("An error occurred: %s", str(e))