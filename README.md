
# Large Langage Model As a Service


<p align="center">
  <img src="Diagram.svg" >
</p>

# LLM-as-a-Service Setup Guide

This guide provides the steps to set up and run the LLM-as-a-Service on your machine.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA A100 or equivalent.
- **Software**: CUDA version 11.8 or above.

## Python Environment Setup

Install required Python packages
```
pip install -r requirements.txt
```
## Configuration

In the cluster_conf.yaml, you must specify access tokens for both your Huggingface and Weights & Biases accounts.

If you wish not to monitor deployment logging, set WANDB_ENABLE to False.
When WANDB_ENABLE is set to False, there's no need to provide a Weights & Biases token.
## Run Weaviate container 

```
sudo docker compose up -d
```

## Running the Inference Service Locally 
Initialize a RAY cluster
```
ray start --head
```
Build the configuration for your backend service (only once).
```
serve build backend:app -o config.yaml
```
Deploy the application
```
serve deploy config.yaml
```

## Running the Inference Service on a Kubernutes Cluster

First you need to set up your KubeRay Cluster. Follow the steps [here](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/raycluster-quick-start.html#kuberay-raycluster-quickstart)

Once you set up the Kubernetes cluster, set the vlaue of `cluster_URL` in cluster_config.yaml.
Submit the job tto your cluster by runing 
```
python job_submission.py
```
Modeify `Ray_service_URL` in cluster_config.yaml to your Kubernetes cluster URL address. 

### Run the authentication microservice
Change the directory to API
```
cd API
```
Run authentication micro-service. 
```
uvicorn app:app --reload --port 8081
```

# Use LLM service. 

## Use Streamlit UI

Run the Streamlit UI 

```
streamlit run UI/main.py
```
For the first time login as admin to add new users.
admin creientials are stored in ```cluster_config.yaml```
More details of streamlit UI can be found [here](UI/README.md)

## Use EscherCloudAI API
To use API service you need to get user access tokens, using streamlit UI, to be able to send request to LLMs API.
You can fine more details of API [here](eschercloud/README.md)