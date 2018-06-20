cd ../../..

./build/tools/caffe train \
--solver=models/MobileNet/SFD-640x640-Quantized/solver.prototxt \
--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNet/SFD-640x640-Quantized/logs/SFD-640x640-Quantized-MobileNetV1-CaffeSSD-Float32-Dense.log

#--weights=models/MobileNet/SSD/mobilenet_iter_73000.caffemodel \

