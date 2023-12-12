
import os
import yaml
from typing import Optional
from sqlalchemy.orm import Session
import weaviate
import logging
import ray
from typing import Any, List
import pypdf
import ray
from ray import serve
import os
import weaviate
from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter
import yaml
from langchain.document_loaders import TextLoader
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from typing import Optional
from pydantic import BaseModel
from backend_database import Database
import os
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)

    
MAX_FILE_SIZE = config.max_file_size * 1024 * 1024  

class VDBaseInput(BaseModel):
    username: Optional[str] 
    class_name: Optional[str] 
    mode: Optional[str]
    vectorDB_type: Optional[str] = "Weaviate"
    file_path: Optional[str] = None



VDB_app = FastAPI()


@ray.remote(num_gpus=config.VD_WeaviateEmbedder_num_gpus, num_cpus=12)
class WeaviateEmbedder:
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
                url=config.weaviate_client_url,   
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
        results= self.text_list
        ray.get(results)
        return self.text_list

    def get(self):
        return self.lst_embeddings
    
    def get_time_taken(self):
        return self.time_taken
    
@serve.deployment(ray_actor_options={"num_gpus": config.VD_deployment_num_gpus}, autoscaling_config={
        #"min_replicas": config.VD_min_replicas,
        "initial_replicas": config.VD_initial_replicas,
        #"max_replicas": config.VD_max_replicas,
        #"max_concurrent_queries": config.VD_max_concurrent_queries,
        })

@serve.ingress(VDB_app)
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
        #text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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

    def process_all_docs(self, dir, username, cls):
        response = {"status": "initiated", "message": ""}
        try:
            full_class = str(username) + "_" + str(cls)
            document_list = self.parse_pdf(dir)
            serialized_docs = self.weaviate_split_multiple_pdf(document_list)
            if len(serialized_docs) <= 160:
                self.add_weaviate_document(full_class, serialized_docs)
                response["status"] = "success"
                response["message"] = f"Processed {len(serialized_docs)} documents for class {full_class}."
            else:
                doc_workload = self.divide_workload(self.num_actors, serialized_docs)
                self.add_weaviate_batch_documents(full_class, doc_workload)
                #self.logger.info(f"check weaviate add data, ")
                response["status"] = "success"
                response["message"] = f"Processed {len(serialized_docs)} documents in batches for class {full_class}."
            return response
        except Exception as e:
            response["status"] = "error"
            response["message"] = str(e)
            return response

    def add_weaviate_document(self, cls, docs):
        actor = WeaviateEmbedder.remote()
        ray.get([actor.adding_weaviate_document.remote(docs, str(cls))])

    def add_weaviate_batch_documents(self, cls, doc_workload):
        actors = [WeaviateEmbedder.remote() for _ in range(3)]
        self.logger.info(f"actors creation successful {actors}: %s", )
        [actor.adding_weaviate_document.remote(doc_part, str(cls)) for actor, doc_part in zip(actors, doc_workload)]
        self.logger.info(f"check 1st step of ray was successful", )
        self.logger.info(f"check if ray was successful:", )


    def add_vdb_class(self,username, class_name,):

        try:            
                weaviate_client = weaviate.Client("http://localhost:8080")
                self.logger.info("checkpoint 1")
                prefix = username
                self.logger.info(f"checkpoint 2 {prefix}: %s",)
                cls = str(prefix) + "_" + str(class_name)
                self.logger.info(f"checkpoint 2 {cls}: %s",)
                #class_description = str(description)
                vectorizer = 'text2vec-transformers'
                if cls is not None:
                    schema = {'classes': [ 
                        {
                                'class': str(cls),
                                'description': 'normal description',
                                'vectorizer': str(vectorizer),
                                'moduleConfig': {
                                    str(vectorizer): {
                                        'vectorizerClassName': False,
                                        }
                                },
                                'properties': [{
                                    'dataType': ['text'],
                                    'description': 'the text from the documents parsed',
                                    'moduleConfig': {
                                        str(vectorizer): {
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
                    weaviate_client.schema.create(schema)
                    database_response = self.database.add_collection({"username": username, "collection_name": class_name})
                    if database_response:
                        self.logger.info("class name added successfully to database")     
                    self.logger.info(f"success: class {class_name} created for user {username}")
                    return {"success": f"Class {cls} created "}
                else:
                    return {"error": "No class name provided"}
        except Exception as e:
            return {"error": str(e)}


    def delete_weaviate_class(self, username, class_name):
            try: 
                weaviate_client = weaviate.Client("http://localhost:8080")
                username = username
                class_name = class_name
                full_class_name = str(username) + "_" + str(class_name)
                if full_class_name is not None:
                    weaviate_client.schema.delete_class(full_class_name)
                    self.database.delete_collection({"username": username, "collection_name": class_name})
                    return {"success": f"Class {full_class_name} has been removed"}
                else:
                    return {"error": "No class name provided"}
            except Exception as e:
                return {"error": str(e)}


    def query_weaviate_document_names(self, username, class_name):
        try:
            weaviate_client = weaviate.Client("http://localhost:8080")
            prefix = username
            prefix = prefix.capitalize()
            class_properties = ["document_title"]
            class_name = class_name
            #full_class_name = str(username) + "_" + str(class_name)
            full_class_name = prefix + "_" + str(class_name)
            query = weaviate_client.query.get(full_class_name, class_properties)
            query = query.do()
            document_title_set = set()
            documents = query.get('data', {}).get('Get', {}).get(str(full_class_name), [])
            for document in documents:
                document_title = document.get('document_title')
                if document_title is not None:
                    document_title_set.add(document_title)
            if document_title_set is not None:
                self.logger.info(f"query success: {len(document_title_set)} documents found")
                return list(document_title_set)
            else:
                return {"error": "No documents found"}
        
        except Exception as e:
                return {"error": str(e)}
        

    def get_classes(self, username):
        try:
            weaviate_client = weaviate.Client("http://localhost:8080")
            username = username
            schema = weaviate_client.schema.get()
            classes = schema.get('classes', []) 
            prefix = str(username) + "_"
            prefix = prefix.capitalize()
            filtered_classes = [cls["class"].replace(prefix, "", 1) for cls in classes if cls["class"].startswith(prefix)] #[cls["class"] for cls in classes if cls["class"].startswith(prefix)]
            if filtered_classes is not None:
                return filtered_classes
            else:
                return {"error": "No classes found"}
        except Exception as e:
                return {"error": str(e)}
        
    @VDB_app.post("/")
    async def VectorDataBase(self, request: VDBaseInput):
            try:
                if request.mode == "add_to_collection":
                    #self.logger.info(f"request received {request}: %s", )
                    response  = self.process_all_docs(request.file_path, request.username, request.class_name)
                    self.logger.info(f"response: {response}: %s", )
                elif request.mode == "display_classes":
                    response = self.get_classes(request.username)
                    self.logger.info(f"classes: {response}: %s", )
                    return response
                elif request.mode == "display_documents":
                    response = self.query_weaviate_document_names(request.username, request.class_name)
                    return response
                elif request.mode == "delete_class":
                    response = self.delete_weaviate_class(request.username, request.class_name)
                    self.logger.info(f"collection delete: {response}: %s", )
                    return response
                elif request.mode == "delete_document":
                    self.delete_weaviate_document(request.data, request.collection_name)
                elif request.mode == "create_collection":
                    self.logger.info(f"checking the request/ {request}: %s", )
                    response = self.add_vdb_class(request.username, request.class_name)
                    return response
                self.logger.info(f"request processed successfully {request}: %s", )
                return {"username": request.username, "response": response}
            except Exception as e:
                self.logger.error("An error occurred: %s", str(e))