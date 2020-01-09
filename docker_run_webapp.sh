#!/bin/bash

echo "Starting Docker..."
docker run --rm \
	${LOCALMOUNT:+ -v $(realpath ~/ncr_model_params):/root/opt/ncr/model_params:ro} \
	-p 127.0.0.1:5000:5000 \
	-e AUTOTEST=$AUTOTEST \
	-e TEST_IGNORE_SCORE=$TEST_IGNORE_SCORE \
	-it ccmsk/neuralcr
