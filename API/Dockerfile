# Use the official Python image
FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# Copy the Python script into the container
COPY . .

# Expose ports
EXPOSE 8088 
EXPOSE 8000
EXPOSE 8082
