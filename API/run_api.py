import uvicorn
import pathlib
#from app.logging_config import setup_logger
import yaml

current_path = pathlib.Path(__file__).parent
config_path = current_path.parent / 'cluster_conf.yaml'
 
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    
API_port = config["API_port"]

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=API_port, reload=True)
