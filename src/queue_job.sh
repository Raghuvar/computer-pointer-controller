#!/bin/bash

exec 1>output/stdout.log 2>output/stderr.log

mkdir -p $1
OUTPUT_FILE=$1
DEVICE=$2
PRECISION=$3
INPUT_FILE=$4
THRESHOLD=$5

if [ "$PRECISION" = "FP32" ]; then
    FACEMODELPATH=../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml
    POSEMODELPATH=../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml
    LANDMARKSMODELPATH=../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml    
    GAZEMODELPATH=../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml        
else
    FACEMODELPATH=../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml
    POSEMODELPATH=../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
    LANDMARKSMODELPATH=../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml    
    GAZEMODELPATH=../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml        
fi

if echo "$DEVICE" | grep -q "FPGA"; then # if device passed in is FPGA, load bitstream to program FPGA
    #Environment variables and compilation for edge compute nodes with FPGAs
    source /opt/intel/init_openvino.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R4_PL1_FP16_MobileNet_Clamp.aocx
fi

python3 main-intel-dev.py -fm ${FACEMODELPATH} \
                -pm ${POSEMODELPATH} \
                -lm ${LANDMARKSMODELPATH} \
                -gm ${GAZEMODELPATH} \
                -i ${INPUT_FILE} \
                -o ${OUTPUT_FILE} \
                -d ${DEVICE} \
                -c ${THRESHOLD}