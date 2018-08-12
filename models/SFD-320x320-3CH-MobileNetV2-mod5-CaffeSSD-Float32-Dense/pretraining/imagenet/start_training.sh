CAFFE_ROOT=$(realpath "../../../../../")

mkdir -p logs snapshots

${CAFFE_ROOT}/build/tools/caffe train \
--solver=solver.prototxt \
--gpu=0,1,2,3 2>&1 | tee logs/MobileNetV2-mod5-ILSVRC12-Dense.log
