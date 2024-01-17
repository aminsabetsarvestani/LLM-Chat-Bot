import json
import requests

backend_url = "http://localhost:8007/stream_chat"
message_content = "Hello, how are you?"
data = {"content": message_content}

headers = {"Content-type": "application/json"}

with requests.post(backend_url, data=json.dumps(data), headers=headers, stream=True) as r:
     for line in r.iter_content():
            print(line.decode("utf-8"), end="")
