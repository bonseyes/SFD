cd ../../..

./build/tools/caffe train \
--solver=models/MobileNet/SFD-640x640-new/solver.prototxt \
--snapshot=models/MobileNet/SFD-640x640-new/snapshots/SFD-MobileNet_640x640_new_WiderFace_iter_65000.solverstate \
--gpu=0 2>&1 | tee models/MobileNet/SFD-640x640-new/logs/SFD-640x640-new-MobileNetV1-CaffeSSD-Float32-Dense-3.log

#--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \

