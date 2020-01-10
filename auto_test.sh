#!/bin/bash
wait-for-it -t 0 127.0.0.1:5000
cd ~/opt/ncr/tests/ci_tests
./test_all.sh || exit -1 
echo "All tests passed!"
