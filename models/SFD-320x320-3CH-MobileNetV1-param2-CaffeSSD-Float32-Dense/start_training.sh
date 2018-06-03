cd ../../..

./build/tools/caffe train \
--solver=models/MobileNet/SFD-320x320-param2/solver.prototxt \
--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNet/SFD-320x320-param2/logs/SFD-320x320-MobileNetV1-param2-CaffeSSD-Float32-Dense.log

#--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \

