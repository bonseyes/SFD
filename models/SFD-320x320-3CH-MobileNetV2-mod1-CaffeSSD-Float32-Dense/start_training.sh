cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SFD-320x320-mod-1/solver.prototxt \
--weights=models/MobileNetV2/MobilenetV2-imagenet_iter_1840000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNetV2/SFD-320x320-mod-1/logs/SSD320x320-MobileNetV2-mod1-CaffeSSD-wider.log

