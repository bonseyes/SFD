cd ../../..

./build/tools/caffe train \
--solver=models/VGGNet/SFD_1CH/solver.prototxt \
--weights=models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
--gpu=0 2>&1 | tee models/VGGNet/SFD_1CH/logs/SFD-640x640-1CH-VGG-CaffeSSD-Float32-Dense.log
