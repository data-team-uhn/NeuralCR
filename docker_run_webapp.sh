#!/bin/bash

echo "Starting Docker..."
docker run --rm \
	-v $(realpath ~/ncr_model_params):/root/opt/ncr/model_params:ro \
	-p 127.0.0.1:5000:5000 \
	-it ccmsk/neuralcr
