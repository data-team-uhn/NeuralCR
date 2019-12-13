#!/bin/bash

HPF_DOMAIN_NAME="hpf.ccm.sickkids.ca"
HPF_USERNAME=$1

echo "Copying ~/ncr_obo_uploads..."
scp -r ~/ncr_obo_uploads/* $HPF_USERNAME@$HPF_DOMAIN_NAME:~/uploaded_obo/

echo "Copying ~/ncr_qsub..."
scp -r ~/ncr_qsub/* $HPF_USERNAME@$HPF_DOMAIN_NAME:~/qsub/
