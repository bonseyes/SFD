export PYTHONPATH=/home/ubuntu/caffe-SSD-COCO/python:/home/ubuntu/caffe-SSD-COCO:$PYTHONPATH
export PATH=/home/ubuntu/caffe-SSD-COCO:$PATH

cd ../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SSD-MobileNetV2-mod5/solver.prototxt \
--snapshot=models/MobileNetV2/SSD-MobileNetV2-mod5/snapshots/SSD-320x320-MobilenetV2-mod5-coco_iter_70000.solverstate \
--gpu=0 2>&1 | tee models/MobileNetV2/SSD-MobileNetV2-mod5/logs/SSD320x320-MobileNetV2-mod5-CaffeSSD-coco-2.log 


