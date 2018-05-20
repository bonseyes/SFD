cd ../../..

./build/tools/caffe train \
--solver=models/MobileNet/SFD-320x320-new/solver.prototxt \
--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNet/SFD-320x320-new/logs/SFD-320x320-new-3CH-MobileNetV1-CaffeSSD-Float32-Dense.log

#--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \

