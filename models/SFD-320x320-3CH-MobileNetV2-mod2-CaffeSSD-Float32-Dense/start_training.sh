cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SFD-320x320-mod-2/solver.prototxt \
--weights=models/MobileNetV2/MobilenetV2-mod2-imagenet_iter_1120000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNetV2/SFD-320x320-mod-2/logs/SSD320x320-MobileNetV2-mod2-CaffeSSD-wider.log

