# Models
This folder contains pre-trained models for SFD using different architectures and training environments.

__Models trained with CaffeSSD__

| Model ID    | Model Name    | Iterations   | Size (MB)   | Speed (ms)  | Accuracy (Easy) | Accuracy (Medium) | Accuracy (Hard) | Download URL |
| ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| --------------|
| A1 | Original                                        | 120K | 86 MB | .. ms | 93.9%| 92.2%| 83.4%| [link](https://drive.google.com/file/d/1CboBIsjcDQ-FC1rMES6IjTl6sYQDoD6u/view) |
| A2 | SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense      | 120K | 86 MB | .. ms | 93.3%| 92.0%| 82.3%| [link](https://drive.google.com/drive/u/0/folders/1WbTmDlUst-90lB8NC_KtcE11v49wW7W2) |
| A3 | SFD-640x640-1CH-VGG-CaffeSSD-Float32-Dense      | 120K | 86 MB | .. ms | 77.4% | 81.1% | 70.5% | [link](https://drive.google.com/open?id=1fudH6TU29F0oOVlIE2trulzCdhmJqJUh) |
| A4 | SFD-640x640-3CH-MobileNetV1-CaffeSSD-Float32-Dense      | 120K | 20 MB | .. ms | 90.0% | 84.8% | 50.1% | [link](https://drive.google.com/open?id=1ROB_Jfjy5PO9V5Xrr72yxqKVWDxNFUkl) |
| A5 | SFD-300x300-3CH-MobileNetV2-CaffeSSD-Float32-Dense      | .. | .. MB | .. ms | .. % | .. % | .. % | .. |

__Models trained with NVIDIA Caffe 0.17__

| Model ID    | Model Name    | Iterations   | Size (MB)   | Speed (ms)  | Accuracy (Easy) | Accuracy (Medium) | Accuracy (Hard) | Download URL |
| ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| --------------|
| B1 | SFD-640x640-3CH-VGG-NVIDIACaffe-Float16-Dense      | .. | -- MB | .. ms | .. % | .. % | .. % | .. |
| B2 | SFD-640x640-3CH-MobileNetV1-NVIDIACaffe-Float16-Dense      | .. | .. MB | .. ms | .. % | .. % | .. % | .. |
| B3 | SFD-300x300-3CH-MobileNetV2-NVIDIACaffe-Float16-Dense      | .. | .. MB | .. ms | .. % | .. % | .. % | .. |

### Architectures

##### VGG 640x640
- [VGG Architecture - 640x640](https://github.com/oylz/SFD/blob/master/model/deploy.prototxt)
- Features Layers : conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2

##### MobileNetV1 640x640
- [MobileNetV1 Architecture - 640x640]()
- [MobileNetV1 Architecture - Pretrained Model COCO](https://drive.google.com/open?id=0B3gersZ2cHIxVFI1Rjd5aDgwOG8)
- Features Layers : conv3, conv5, conv11, conv13, conv14_2, conv15_2

##### MobileNetV1 300x300
- [MobileNetV1 Architecture - 300x300 Protoxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.prototxt)
- [MobileNetV1 Architecture - Pretrained Model COCO + VOC0712](https://drive.google.com/open?id=0B3gersZ2cHIxVFI1Rjd5aDgwOG8)
- Features Layers : __To decide__

##### MobileNetV2 300x300
- [MobileNetV2 Architecture - 300x300 Protoxt](https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/train.prototxt)
- Features Layers : __To decide__


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
3. MobileNetV2

__Caffe Versions__
1. [Caffe SSD](https://github.com/weiliu89/caffe/tree/ssd) : CaffeSSD
2. [NVIDIA Caffe 0.17](https://github.com/NVIDIA/caffe) : NVIDIACaffe
3. [TIDSP Jacinto Caffe 0.16](https://github.com/tidsp/caffe-jacinto) : TIDSPCaffe
4. [Bonseyes Jacinto Caffe 0.16](https://github.com/bonseyes/caffe-jacinto) : BonseyesCaffe

__Precision__
1. Float32
2. Float16

__Training Methods__
1. Dense
2. Sparse
3. Quantized



