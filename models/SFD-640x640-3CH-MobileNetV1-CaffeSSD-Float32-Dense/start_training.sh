cd ../../..

./build/tools/caffe train \
--solver=models/MobileNet/SFD_3CH/solver.prototxt \
--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNet/SFD_3CH/logs/SFD-640x640-3CH-MobileNetV1-CaffeSSD-Float32-Dense.log

#--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \

