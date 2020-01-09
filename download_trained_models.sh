#!/bin/bash
echo "Downloading the trained models from GitHub..."
cd /root/opt/ncr/
mkdir model_params
cd model_params
wget https://github.com/ccmbioinfo/NeuralCR/releases/download/1.0/ncr_model_params.tar.gz
tar -xvf ncr_model_params.tar.gz
rm ncr_model_params.tar.gz
cp -v -r ncr_model_params/* .
rm -r ncr_model_params
echo "Finished installing trained models"
