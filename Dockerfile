# Start from a small Python3 base image
# Python 3.6 is required since newer versions aren't supported by the old version of tensorflow in use
FROM python:3.6-slim-buster

# Install gcc, needed to build fasttext
RUN apt-get update && apt-get install -y g++ wget wait-for-it

# Install dependencies
RUN pip3 install 'cython==0.29.14' 'scipy==1.4.0' 'tensorflow==1.13.2' 'fasttext==0.9.1' 'Flask==1.1.1' 'Orange-Bioinformatics==2.6.25'

# Put everything in /opt/ncr
RUN mkdir -p /root/opt/ncr/
WORKDIR /root/opt/ncr/
COPY . ./

# This is the default command executed when starting the container
COPY startup_script.sh /
RUN chmod u+x /startup_script.sh
COPY download_trained_models.sh /
RUN chmod u+x /download_trained_models.sh
COPY auto_test.sh /
RUN chmod u+x /auto_test.sh
ENTRYPOINT /startup_script.sh
