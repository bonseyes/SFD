
# SFD-640x640-3CH-ResNetV1-CaffeSSD-Float32-Dense Model

__Input__
+ 640x640
+ 3 channels RGB

__Architecture__
+ MobileNet-V1
+ [MobileNetV1 Architecture - Pretrained Model COCO](https://drive.google.com/open?id=0B3gersZ2cHIxVFI1Rjd5aDgwOG8)
+ Features Layers : conv5, conv11, conv13, conv14_2, conv15_2

__Training__
+ Caffe SSD
+ Float32
+ Dense
+ Mean values: [104, 117, 123]

__Files__
+ [deploy](deploy.prototxt)
+ [train](train.prototxt)
+ [test](test.prototxt)
+ [trained model 100K iterations]()

__Results[100K iterations]__
+ Easy: 90.4%
+ Medium: 85.6%
+ hard: 66.6%

__Notes__
+ This model used different bounding box parameters (param2)
+ In previous confogurations, it is noticable that accuracy for small faces (hard subset) is not high, this is mainly because the first feature map is very early and not good enough for detection. In this update, "conv3" which handle the detection of bounding boxes with size 8x8 (stride 4) is not used for detection, instead "conv5" is used for boxes of sizes 8x8 and 16x16 with stride of 8 for both of them. The results shows better accuracy for all the 3 splits (hard, medium, and easy)   

