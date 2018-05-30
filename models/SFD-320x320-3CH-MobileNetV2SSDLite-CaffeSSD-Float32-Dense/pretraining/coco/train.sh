#!/bin/sh
mkdir -p snapshot logs
# If you are using ReLU6 on your train.prototxt, better use this version of Caffe: https://github.com/chuanqi305/ssd
# Alternatively, you can replace ReLU6 by regular ReLUs, but that will hinder final results
../../../../../../caffe-ssd-relu6/build/tools/caffe train -solver="models/solver_train.prototxt" 2>&1 | tee "./logs/$(date +%Y%d%m%H%M%S)SFD-320x320-3CH-MobileNetV2SSDLite-CaffeSSD-Float32-Dense.log"
# -weights="deploy.caffemodel" \
# -gpu 0
