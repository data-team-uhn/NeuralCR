#!/bin/bash

echo "Starting Docker..."
docker run --rm \
	-p 127.0.0.1:5000:5000 \
	-e AUTOTEST=$AUTOTEST \
	-it ccmsk/neuralcr
