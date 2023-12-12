from ray.job_submission import JobSubmissionClient
import yaml
import os

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)


# Get the current working directory
current_directory = os.getcwd()

runtime_env = {"working_dir": f"{current_directory}", "pip":  [
        "langchain",
        "fastapi",
        "wandb",
        "ray[data,train,tune,serve]",
        "eschercloud",
        "SQLAlchemy",
        "passlib",
        "-f https://download.pytorch.org/whl/cu118 torch",
        "-f https://download.pytorch.org/whl/cu118 torchvision",
        "-f https://download.pytorch.org/whl/cu118 torchaudio",
        "xformers",
        "accelerate",
        "sentencepiece",
        "bitsandbytes",
        "einops",
        "streamlit",
        "python-dotenv",
        "transformers",
        "scipy"
    ]}
client = JobSubmissionClient(config.cluster_URL)
job_id = client.submit_job(
    entrypoint="python backend.py",
    runtime_env=runtime_env,
)