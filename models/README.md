# Models
This folder contains pre-trained models for SFD using different architectures and training environments.

| Model Name    | Size (MB)   | Speed (ms)  | Accuracy Easy | Accuracy Medium | Accuracy Hard | Download URL |
| ------------- |-------------|-------------| --------------| ----------------| --------------| --------------|
| Original                                        | xxMB | 130ms | 92.3% | 90.3% | 83.3% | [Download](https://www.google.com) |
| SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense      | xxMB | 130ms | 92.3% | 90.3% | 83.3% | [Download](https://www.google.com) |
| SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense      | xxMB | 130ms | 92.3% | 90.3% | 83.3% | [Download](https://www.google.com) |

### Naming Convention
The following naming convention is used for models:

+ SFD-{Input}-{Arch}-{CaffeVersion}-{Precision}-{Method}

e.g. SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense

__Input__
1. 640x640 RGB 3 Channel : 640x640-3CH
2. 640x640 Gray 1 Channel : 640x640-1CH

__Architectures__
1. VGG
2. MobileNetV1
3. MobileNetV2

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



