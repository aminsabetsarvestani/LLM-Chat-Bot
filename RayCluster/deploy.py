from backend_hfTextGen import PredictDeployment
#from backend_vectordatabase import VectorDataBase
#from backend_arxiv_search import ArxivSearch
from ray import serve
import yaml
import ray



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)
# Deploy the vector database

vdb_deployment_options = {
    
}

ray.init() 
serve.start()
#serve.run(VectorDataBase.bind(), name = "VectorDB" ,route_prefix="/VectorDB")
#serve.run(ArxivSearch.bind(), name = "ArxivSearch" ,route_prefix="/ArxivSearch")

for LLM in config.LLMs:
    print("--------------->  Deploying LLMs")
    prefix = LLM["route_prefix"]
    
    
    # Define the deployment with the specified configurations from LLM
    deployment_options = {
        "name": LLM["name"],
        "route_prefix": f"/{prefix}",
        "ray_actor_options": {"num_gpus": LLM["num_gpus"]},
        "autoscaling_config": {
            "min_replicas": LLM["min_replicas"],
            "initial_replicas": LLM["initial_replicas"],
            "max_replicas": LLM["max_replicas"],
            "target_num_ongoing_requests_per_replica": LLM["target_num_ongoing_requests_per_replica"],
            "graceful_shutdown_wait_loop_s": LLM["graceful_shutdown_wait_loop_s"],
            "max_concurrent_queries": LLM["max_concurrent_queries"],
        }
    }
     # Apply the deployment options to the PredictDeployment class
    PredictDeployment = PredictDeployment.options(**deployment_options)
    #Deploy the service with the specified model_id and other parameters
    serve.run(PredictDeployment.bind(LLM["model_tokenizer"], LLM["temperature"], LLM["max_new_tokens"], LLM["repetition_penalty"], LLM["batch_size"]), name = LLM["name"], route_prefix=f"/{prefix}")