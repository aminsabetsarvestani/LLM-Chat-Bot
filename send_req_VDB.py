import requests

# Your existing parameters
param = {
  "class_name": "sdee",
  "mode": "create_collection",
  "vectorDB_type": "Weaviate",
  "file_path": "string"
}

# File to be sent
file_to_send = {
    'file': open('docs/paper.pdf', 'rb')  # Replace 'path_to_your_file' with the actual file path
}

# Token for authorization header
nils_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWluIiwiZXhwIjoxNzAyMDQ4OTAyfQ.JrG5HMPxFbeS_EI30GQD17XGfyTx2K43BU5wRrXoIQY"
url = 'http://localhost:8083/vector_DB_request/'

# Sending the request
response = requests.post(url, data=param, files=file_to_send, headers={'Authorization': f"Bearer {nils_token}"})

# Close the file after sending
file_to_send['file'].close()

# Print response
print(response.status_code)
print(response.json())
