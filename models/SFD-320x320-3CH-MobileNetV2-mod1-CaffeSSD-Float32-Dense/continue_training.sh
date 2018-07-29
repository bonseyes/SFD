cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SFD-320x320-mod-1/solver.prototxt \
--snapshot=models/MobileNetV2/SFD-320x320-mod-1/snapshots/SFD-320x320-MobilenetV2-mod1-wider_iter_45000.solverstate \
--gpu=0 2>&1 | tee models/MobileNetV2/SFD-320x320-mod-1/logs/SSD320x320-MobileNetV2-mod1-CaffeSSD-wider-2.log 


