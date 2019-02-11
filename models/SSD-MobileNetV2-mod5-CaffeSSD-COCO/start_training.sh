export PYTHONPATH=/home/ubuntu/caffe-SSD-COCO/python:/home/ubuntu/caffe-SSD-COCO:$PYTHONPATH
export PATH=/home/ubuntu/caffe-SSD-COCO:$PATH

cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SSD-MobileNetV2-mod5/solver.prototxt \
--weights=models/MobileNetV2/MobilenetV2-mod5-imagenet_iter_570000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNetV2/SSD-MobileNetV2-mod5/logs/SSD320x320-MobileNetV2-mod5-CaffeSSD-coco.log

#--weights=models/MobileNetV2/MobilenetV2-mod2-imagenet_iter_1120000.caffemodel \


