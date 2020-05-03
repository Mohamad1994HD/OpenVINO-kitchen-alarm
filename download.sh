#!/bin/bash

source /opt/intel/openvino/bin/setupvars.sh

mkdir models

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --list models.lst -o ./models
