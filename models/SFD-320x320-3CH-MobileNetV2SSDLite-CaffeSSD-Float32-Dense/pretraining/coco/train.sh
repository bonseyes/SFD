#!/bin/sh
mkdir -p snapshot
../../../../../build/tools/caffe train -solver="models/solver_train.prototxt" \
# -weights="deploy.caffemodel" \
-gpu 0
