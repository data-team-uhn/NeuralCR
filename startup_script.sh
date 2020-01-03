#!/bin/bash
[ -d /root/opt/ncr/model_params ] || /download_trained_models.sh
[ -z "$AUTOTEST" ] || { python3 app.py & }
[ -z "$AUTOTEST" ] || exec /auto_test.sh
[ -z "$AUTOTEST" ] && python3 app.py
