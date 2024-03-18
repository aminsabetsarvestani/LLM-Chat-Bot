import ray
from ray import serve
import torch
import time
import requests
ray.init(num_gpus=1)

@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class GPUTask:
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def __call__(self, request):
        
        tensor = torch.ones(10000, 10000, device=self.device)
        # Perform some computations (simulating a long-running task)
        for _ in range(10):
            tensor = torch.matmul(tensor, tensor)
        return "Task completed on GPU"



serve.run(GPUTask.bind(), route_prefix="/gpu_task")

gpu_task_url = "http://127.0.0.1:8000/gpu_task"

# Make a request to the GPU task
response = requests.get(gpu_task_url)
print(response.text)