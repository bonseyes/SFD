# SFD-640x640-3CH-MobileNetV1--Quantized-CaffeSSD-Float32-Dense Model

__Input__
+ 640x640
+ 3 channels RGB

__Architecture__
+ MobileNet-V1 quantization friendly
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
+ [trained model 100K iterations](https://drive.google.com/open?id=1bE5r4OqOicsm8csJOftmP0HRMzFhZVGy)

__Results[100K iterations]__
+ Easy: 87.5%
+ Medium: 82.8%
+ hard: 63.4%

__Notes__
+ source: https://arxiv.org/pdf/1803.08607.pdf   

