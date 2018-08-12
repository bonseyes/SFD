cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SFD-640x640-mod-4/solver.prototxt \
--weights=models/MobileNetV2/MobilenetV2-mod2-imagenet_iter_1120000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNetV2/SFD-640x640-mod-4/logs/SSD640x640-MobileNetV2-mod4-CaffeSSD-wider.log

