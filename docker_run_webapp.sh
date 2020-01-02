#!/bin/bash

mkdir ~/ncr_obo_uploads || { echo "~/ncr_obo_uploads ALREADY EXISTS"; exit -1; }
mkdir ~/ncr_qsub || { echo "~/ncr_qsub ALREADY EXISTS"; exit -1; }

echo "Starting Docker..."
docker run --rm \
	-v $(realpath ~/ncr_obo_uploads):/root/uploaded_obo:rw \
	-v $(realpath ~/ncr_qsub):/root/qsub:rw \
	-v $(realpath ~/ncr_model_params):/root/opt/ncr/model_params:ro \
	-p 127.0.0.1:5000:5000 \
	-it ccmsk/neuralcr
