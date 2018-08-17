CAFFE_ROOT=$(realpath "../../../../../")

mkdir -p logs snapshots

${CAFFE_ROOT}/build/tools/caffe train \
--solver=solver.prototxt \
--gpu=2 2>&1 | tee logs/MobileNetV2-mod4-ILSVRC12-Dense.log
