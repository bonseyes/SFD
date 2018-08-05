cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SFD-320x320-mod-5/solver.prototxt \
--gpu=0 2>&1 | tee models/MobileNetV2/SFD-320x320-mod-5/logs/SSD320x320-MobileNetV2-mod5-CaffeSSD-wider.log

#--weights=models/MobileNetV2/MobilenetV2-mod2-imagenet_iter_1120000.caffemodel \


