# Start from a small Python3 base image
# Python 3.6 is required since newer versions aren't supported by the old version of tensorflow in use
FROM python:3.6-slim-buster

# Install gcc, needed to build fasttext
RUN apt-get update && apt-get install -y g++

# Install dependencies
RUN pip3 install 'cython' 'scipy' 'tensorflow==1.13.2' 'fasttext==0.9.1' 'Flask==1.1.1'

# Put everything in /opt/ncr
RUN mkdir -p /opt/ncr/
WORKDIR /opt/ncr/
COPY * ./

# This is the default command executed when starting the container
ENTRYPOINT python3 app.py