cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SFD-640x640-mod-2/solver.prototxt \
--weights=models/MobileNetV2/deploy.caffemodel \
--gpu=0 2>&1 | tee models/MobileNetV2/SFD-640x640-mod-2/logs/SSD640x640-MobileNetV2-mod2-CaffeSSD-wider.log

