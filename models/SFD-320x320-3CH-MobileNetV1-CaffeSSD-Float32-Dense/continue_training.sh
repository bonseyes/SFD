cd ../../..

./build/tools/caffe train \
--solver=models/MobileNet/SFD-320x320-new/solver.prototxt \
--snapshot=/home/ubuntu/SSD/models/MobileNet/SFD-320x320-new/snapshots/SFD-MobileNet_320x320_WiderFace_iter_100000.solverstate \
--gpu=0 2>&1 | tee models/MobileNet/SFD-320x320-new/logs/SFD-320x320-new-3CH-MobileNetV1-CaffeSSD-Float32-Dense.log 

#--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \

