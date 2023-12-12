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



#@ray.remote(num_gpus=config.VD_WeaviateEmbedder_num_gpus)
class WeaviateEmbedder:
    def __init__(self):
        self.time_taken = 0
        self.text_list = []
        self.weaviate_client = weaviate.Client(
            url=config.weaviate_client_url,   
        )

    def adding_weaviate_document(self, text_lst, collection_name):
        start_time = time.time()
        self.weaviate_client.batch.configure(batch_size=50)

        with self.weaviate_client.batch as batch:
            for text in text_lst:
                    batch.add_data_object(
                        text,
                        class_name=collection_name, 
                        #uuid=generate_uuid5(text),
        )
        self.text_list.append(text)
        self.time_taken = time.time() - start_time
        return self.text_list

    def get(self):
        return self.lst_embeddings
    
    def get_time_taken(self):
        return self.time_taken
    
MAX_FILE_SIZE = config.max_file_size * 1024 * 1024  

class VDBaseInput(BaseModel):
    username: str 
    collection_name: Optional[str] 
    mode: str = "add_to_collection"
    vectorDB_type: Optional[str] = "Weaviate"
    file_path: Optional[str] = None



VDB_app = FastAPI()

class VectorDataBase:
    def __init__(self):

        self.weaviate_client = weaviate.Client(
            url=config.weaviate_client_url,   
        )
        self.weaviate_vectorstore = Weaviate(self.weaviate_client, 'Chatbot', 'page_content', attributes=['page_content'])
        self.num_actors = config.VD_number_actors
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

    def weaviate_serialize_document(self,doc):
        document_title = doc.metadata.get('source', '').split('/')[-1]
        return {
            "page_content": doc.page_content,
            "document_title": document_title,
        }
    
    def weaviate_split_multiple_pdf(self,docs):    
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        text_docs = text_splitter.split_documents(docs)

        serialized_docs = [
                    self.weaviate_serialize_document(doc) 
                    for doc in text_docs
                    ]
       
        return serialized_docs	

    def divide_workload(self, num_actors, documents):
        docs_per_actor = len(documents) // num_actors

        doc_parts = [documents[i * docs_per_actor: (i + 1) * docs_per_actor] for i in range(num_actors)]

        if len(documents) % num_actors:
            doc_parts[-1].extend(documents[num_actors * docs_per_actor:])

        return doc_parts

    def parse_pdf(self, directory):    
        documents = []
        for file in os.listdir(directory):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(directory, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                except pypdf.errors.PdfStreamError as e:
                    print(f"Skipping file {file} due to error: {e}")
                    continue  # Skip this file and continue with the next one
            elif file.endswith('.txt'):
                text_path = os.path.join(directory, file)
                try:
                    loader = TextLoader(text_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error in file {file}: {e}")
                    continue
        return documents

    def process_all_docs(self, dir, cls):
        document_list = self.parse_pdf(dir)
        serialized_docs = self.weaviate_split_multiple_pdf(document_list)
        if len(serialized_docs) <= 50:
            self.add_weaviate_document(cls, serialized_docs)
        else:
            doc_workload = self.divide_workload(self.num_actors, serialized_docs)
            self.add_weaviate_batch_documents(cls, doc_workload)

    def add_weaviate_document(self, cls, docs):
        actor = WeaviateEmbedder()
        [actor.adding_weaviate_document(doc_part, str(cls)) for doc_part in docs]

    def add_weaviate_batch_documents(self, cls, doc_workload):
        actors = [WeaviateEmbedder() for _ in range(3)]
        [actor.adding_weaviate_document(doc_part, str(cls)) for actor, doc_part in zip(actors, doc_workload)]
        [actor.get_time_taken() for actor in actors]

    def query_weaviate_document_names(self,cls):
        class_properties = ["document_title"]
        query = self.weaviate_client.query.get(cls, class_properties)
        query = query.do()

        document_title_set = set()
        documents = query.get('data', {}).get('Get', {}).get(str(cls), [])

        for document in documents:
            document_title = document.get('document_title')
            if document_title is not None:
                document_title_set.add(document_title)
        return list(document_title_set)
    
    def delete_weaviate_document(self, name, cls_name):
        document_name = str(name)
        self.weaviate_client.batch.delete_objects(
            class_name=cls_name,
            where={
                "path": ["document_title"],
                "operator": "Like",
                "valueText": document_name,
            }
        )

    def delete_weaviate_class(self, name):
        class_name = name
        self.weaviate_client.schema.delete_class(class_name)

    def create_weaviate_class(self, name, embedding_model = "text2vec-transformers"):
        class_name = str(name)
        #class_description = str(description)
        vectorizer = str(embedding_model)
        
        schema = {'classes': [ 
            {
                    'class': class_name,
                    'description': 'normal description',
                    'vectorizer': vectorizer,
                    'moduleConfig': {
                        vectorizer: {
                            'vectorizerClassName': False,
                            }
                    },
                    'properties': [{
                        'dataType': ['text'],
                        'description': 'the text from the documents parsed',
                        'moduleConfig': {
                            vectorizer: {
                                'skip': False,
                                'vectorizePropertyName': False,
                                }
                        },
                        'name': 'page_content',
                    },
                    {
                        'name': 'document_title',
                        'dataType': ['text'],
                    }],      
                    },
        ]}
        self.weaviate_client.schema.create(schema)
    async def process_pdf_file(self, file_data: bytes):
        # Process the PDF file
        pass

    async def extract_and_process_zip(self, file_data: bytes):
        try:
            with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_ref:
                # Extract ZIP file
                tmp_dir = 'temp_dir'
                os.makedirs(tmp_dir, exist_ok=True)
                zip_ref.extractall(tmp_dir)

                # Process each file in the ZIP
                for filename in os.listdir(tmp_dir):
                    if filename.endswith('.pdf'):
                        file_path = os.path.join(tmp_dir, filename)
                        with open(file_path, 'rb') as pdf_file:
                            self.process_pdf_file(pdf_file.read())

                # Clean up temporary directory
                shutil.rmtree(tmp_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")
my_VDB = VectorDataBase()
@VDB_app.post("/")
async def VectorDataBase( request: VDBaseInput):
        try:
            if request.mode == "add_to_collection":
                print(request.file_path)
                response  = my_VDB.process_all_docs(request.file_path, request.collection_name)
                return {"response": response}
            elif request.mode == "query_collection":
                response = my_VDB.query_weaviate_document_names(request.collection_name)
                return {"response": response}
            elif request.mode == "delete_collection":
                my_VDB.delete_weaviate_class(request.collection_name)
            elif request.mode == "delete_document":
                my_VDB.delete_weaviate_document(request.data, request.collection_name)
            elif request.mode == "create_collection":
                response = my_VDB.database.add_collection({"username": request.username, "collection_name": request.collection_name})
                collection_name = response["collection_name"]
                my_VDB.create_weaviate_class(collection_name)
                # request.vectorDB_type needs to be replace with a variable read from config file
            #self.logger.info(f"request processed successfully {request}: %s", )
            return {"response": collection_name}
        except Exception as e:
            #self.logger.error("An error occurred: %s", str(e))
            print(f"An error occurred: {str(e)}")


#serve.run(VectorDataBase.bind(), route_prefix="/")

# Note: remember Weaviate put capital letter for first letter of the collection name 