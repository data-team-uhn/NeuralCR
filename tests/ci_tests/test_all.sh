#!/bin/bash

#Data driven tests
START_DIR=$(pwd)
for ddt in data_driven/*
do
	echo "Running test $ddt"
	cd $ddt
	python3 $START_DIR/../test_template/data_driven_api_test.py || exit -1
	echo "PASS"
	cd $START_DIR
done
