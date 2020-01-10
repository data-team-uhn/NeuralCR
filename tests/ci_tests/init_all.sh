#!/bin/bash

#Data driven tests
START_DIR=$(pwd)
for ddt in data_driven/*
do
	echo "Initializing test $ddt"
	cd $ddt
	python3 $START_DIR/../test_template/initialize_data_driven_api_result.py && echo "Initialized"
	cd $START_DIR
done
