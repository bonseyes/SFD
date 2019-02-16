export PATH=/home/ubuntu/SSD:$PATH
export PYTHONPATH=/home/ubuntu/SSD/python:/home/ubuntu/SSD:$PYTHONPATH

CAFFE_ROOT=$(realpath "../../../../")

mkdir -p logs snapshots

#${CAFFE_ROOT}/build/tools/caffe train \
#cd ../../../..

#./build/tools/caffe train \
${CAFFE_ROOT}/build/tools/caffe train \
--solver=solver.prototxt \
--gpu=0 2>&1 | tee logs/MobileNetV2-mod11l-ImageNetILSVRC12-CaffeSSD.log
