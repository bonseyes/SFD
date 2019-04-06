export PATH=/home/ubuntu/caffe-SSD:$PATH
export PYTHONPATH=/home/ubuntu/caffe-SSD/python:/home/ubuntu/caffe-SSD:$PYTHONPATH

CAFFE_ROOT=$(realpath "../../../../")

mkdir -p logs snapshots

#${CAFFE_ROOT}/build/tools/caffe train \
#cd ../../../..

#./build/tools/caffe train \
${CAFFE_ROOT}/build/tools/caffe train \
--solver=solver.prototxt \
--gpu=0 2>&1 | tee logs/MobileNetV2-mod11e-ImageNetILSVRC12-CaffeSSD.log
