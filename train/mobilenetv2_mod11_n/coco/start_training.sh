export PATH=/home/ubuntu/caffe-NVIDIA:$PATH
export PYTHONPATH=/home/ubuntu/caffe-NVIDIA/python:/home/ubuntu/caffe-NVIDIA:$PYTHONPATH

mkdir -p snapshots
mkdir -p logs

cd ../../../..

./build/tools/caffe train \
--solver=models/MobileNetV2/SFD-320x320-mod11n-fp16/COCO/solver.prototxt \
--weights=/home/ubuntu/caffe-SSD/models/pretrain-MobileNetV2-ImageNet/mod-11/MobileNetV2-320x320-mod11n/snapshots/MobilenetV2-mod11n-ImageNet_iter_675000.caffemodel \
--gpu=0 2>&1 | tee models/MobileNetV2/SFD-320x320-mod11n-fp16/COCO/logs/SSD320x320-MobileNetV2-mod11n-NVCaffe-fp16-coco.log

#--weights=/home/ubuntu/caffe-NVIDIA/models/MobileNetV2/SFD-person+face-320x320-mod5-fp16/snapshots/SFD-person+face-320x320-mod5-fp16-pretrained_iter_150000.caffemodel \
#--weights=/home/ubuntu/caffe-SSD/models/MobileNetV2/MobilenetV2-mod5-imagenet_iter_800000.caffemodel \
#--weights=models/MobileNetV2/MobilenetV2-mod2-imagenet_iter_1120000.caffemodel \


