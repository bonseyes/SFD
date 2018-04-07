# Models
This folder contains pre-trained models for SFD using different architectures and training environments.

| Model Name    | Size (MB)   | Speed (ms)  | Accuracy (Easy) | Accuracy (Medium) | Accuracy (Hard) | Download URL |
| ------------- |-------------|-------------| --------------| ----------------| --------------| --------------|
| Original                                        | .. MB | .. ms | .. % | .. % | .. % | [Download](https://www.google.com) |
| SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense      | .. MB | .. ms | .. % | .. % | .. % | [Download](https://www.google.com) |
| SFD-640x640-1CH-VGG-CaffeSSD-Float32-Dense      | .. MB | .. ms | .. % | .. % | .. % | [Download](https://www.google.com) |
| SFD-300x300-3CH-MobileNetV1-CaffeSSD-Float32-Dense      | .. MB | .. ms | .. % | .. % | .. % | [Download](https://www.google.com) |
| SFD-300x300-1CH-MobileNetV1-CaffeSSD-Float32-Dense      | .. MB | .. ms | .. % | .. % | .. % | [Download](https://www.google.com) |

### Architectures

##### VGG
[VGG Architecture - 640x640 Protoxt](https://github.com/oylz/SFD/blob/master/model/deploy.prototxt)
Features Layers : conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2

##### MobileNetV1
[MobileNetV1 Architecture - 300x300 Protoxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.prototxt)
[MobileNetV1 Architecture - 300x300 Pretrained Model COCO + VOC0712](https://drive.google.com/open?id=0B3gersZ2cHIxVFI1Rjd5aDgwOG8)
Features Layers : __To decide__

### Naming Convention
The following naming convention is used for models:

+ SFD-{Input}-{Arch}-{CaffeVersion}-{Precision}-{Method}

e.g. SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense

__Input__
1. 640x640 RGB 3 Channel : 640x640-3CH
2. 640x640 Gray 1 Channel : 640x640-1CH
3. 300x300 RGB 3 Channel : 300x300-3CH
4. 300x300 Gray 1 Channel : 300x300-1CH

__Architectures__
1. VGG
2. MobileNetV1

__Caffe Versions__
1. Caffe SSD : CaffeSSD [Link]
2. NVIDIA Caffe 0.17 : NVIDIACaffe0.17 [Link]
3. TIDSP Jacinto Caffe 0.16 : TIDSPJacinto0.16 [Link]
4. Bonseyes Caffe 0.16 : BonseyesCaffe0.16 [Link]

__Precision__
1. Float32
2. Float16

__Training Methods__
1. Dense
2. Sparse
3. Quantized



