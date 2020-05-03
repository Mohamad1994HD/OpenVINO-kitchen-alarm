#!/bin/bash

###########################
# Configure here
DETECTOR_MODEL_PATH=./models/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml
# rtsp or webcam (0) or video file
INPUT_FILE=0
ACCELERATOR=CPU
###########################

### Do not change here

source /opt/intel/openvino/bin/setupvars.sh


python3 main.py \
	-m $DETECTOR_MODEL_PATH\
	-i $INPUT_FILE\
	--device $ACCELERATOR
