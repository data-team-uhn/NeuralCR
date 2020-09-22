#!/bin/bash

echo "Starting Docker..."
docker run --rm \
	${LOCALMOUNT:+ -v $(realpath ~/ncr_model_params):/root/opt/ncr/model_params:ro} \
	-v $(realpath ~/webdav_cert.pem):/root/webdav_cert.pem:ro \
	-p 127.0.0.1:5000:5000 \
	-e AUTOTEST=$AUTOTEST \
	-e TEST_IGNORE_SCORE=$TEST_IGNORE_SCORE \
	-e WEBDAV_CERTPATH=/root/webdav_cert.pem \
	-e WEBDAV_APIKEY=$WEBDAV_APIKEY \
	-e QSUB_WEBDAV_URL=$QSUB_WEBDAV_URL \
	-e OBO_WEBDAV_URL=$OBO_WEBDAV_URL \
	-e LOGGING_WEBDAV_URL=$LOGGING_WEBDAV_URL \
	-e OUTPUT_WEBDAV_URL=$OUTPUT_WEBDAV_URL \
	-e READY_WEBDAV_URL=$READY_WEBDAV_URL \
	-e COMPLETE_WEBDAV_URL=$COMPLETE_WEBDAV_URL \
	-e FAILED_WEBDAV_URL=$FAILED_WEBDAV_URL \
	-it ccmsk/neuralcr
