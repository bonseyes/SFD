# SFD Models
This folder contains pre-trained models for SFD using different architectures and training environments.

__Models trained with CaffeSSD__

| Model ID    | Model Name    | Iterations   | Size (MB)   | Speed (ms)  | Wider Accuracy (E/M/H) | AFW Accuracy | Pascal Accuracy | FDDB Accuracy (D/C) | UFDD Accuracy | Download URL | Pretraining (Imagenet) Model |
| ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| --------------| --------------| --------------| --------------|
| A1 | Original                                        | 120K | 86 MB | 31.24 ms | 93.9% / 92.2% / 83.4%| 99.86% | 98.49% | 98.2% / 75.72% | 72.2% | [link](https://drive.google.com/file/d/1CboBIsjcDQ-FC1rMES6IjTl6sYQDoD6u/view) | |
| A2 | SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense      | 120K | 86 MB | 30.28 ms | 93.4% / 92.0% / 82.3%| 99.45% | 97.71% | 97.8% / 74.99% | 69.9% | [link](https://drive.google.com/drive/u/0/folders/1WbTmDlUst-90lB8NC_KtcE11v49wW7W2) | |
| A3 | SFD-640x640-1CH-VGG-CaffeSSD-Float32-Dense      | 120K | 86 MB | 30.29 ms | 77.4% / 81.1% / 70.5% | 98.93% | 97.24% | 95.9% / 73.5% | 44.9% | [link](https://drive.google.com/open?id=1fudH6TU29F0oOVlIE2trulzCdhmJqJUh) | |
| A4 | SFD-640x640-3CH-MobileNetV1-CaffeSSD-Float32-Dense      | 120K | 20 MB | 36.86 ms | 90.0% / 84.8% / 50.1% | 98.36% | 97.27% | 92.6% / 70.83% | 49.7% | [link](https://drive.google.com/open?id=1ROB_Jfjy5PO9V5Xrr72yxqKVWDxNFUkl) | |
| A5 | SFD-320x320-3CH-MobileNetV1-CaffeSSD-Float32-Dense      | 120K | 20 MB | 21.34 ms | 84.3% / 79.1% / 55.3% | 97.45% | 96.04% | 90.0% / 68.42% | 43.9% | [link](https://drive.google.com/drive/folders/1nrcekK5sLOUL3zVCzBaORwZTKw6mXnxM?usp=sharing) | |
| A6 | SFD-640x640-3CH-MobileNetV1-param2-CaffeSSD-Float32-Dense      | 100K | 20 MB | 45.05 ms | 90.4% / 85.6% / 66.6% | 98.40% | 97.49% | 94.38% / 72.23% | 56.1% | [link](https://drive.google.com/open?id=1bf4Y0zcjmjKcxZttd-atGl3ifJzUIgfD) | |
| A7 | SFD-320x320-3CH-MobileNetV1-param2-CaffeSSD-Float32-Dense      | 100K | 20 MB | 36.80 ms | 86.9% / 82.5% / 65.5% | 98.16% | 96.64% | 92.0% / 70.1% | 51.1% | [link](https://drive.google.com/open?id=1_tmrTB0HYzSq3gFPKUWlfKtHQn6WIWNM) | |
| A8 | SFD-640x640-3CH-MobileNetV2-CaffeSSD-Float32-Dense      | 30K | 12.4 MB | 104.21 ms | 90.0% / 87.3% / 72.6% | 99.21% | 98.15% | 88.50% / 67.46% | 59.3% | [link](https://github.com/bonseyes/SFD/tree/master/models/SFD-640x640-3CH-MobileNetV2-CaffeSSD-Float32-Dense) | |
| A9 | SFD-640x640-3CH-MobileNetV2-CaffeSSD-Float32-Dense      | 100K | 12.4 MB | 104.87 ms | --% / --% / --% | 85.60% | 74.12% | --% / --% | --% | [link](https://github.com/bonseyes/SFD/tree/master/models/SFD-640x640-3CH-MobileNetV2-CaffeSSD-Float32-Dense) | |
| A10 | SFD-640x640-Quantized-MobileNetV1-CaffeSSD-Float32-Dense      | 100K | 20 MB | 30.80 ms | 87.5% / 82.8% / 62.4% | --% | --% | --% / --% | --% | [link](https://drive.google.com/open?id=1MtSQoj09DQFYYhVx7Kh4fp7vLm0Z9JQ9) | |
| A11 | SFD-320x320-3CH-MobileNetV2-mod1-CaffeSSD-Float32-Dense   | 90K | 12 MB | 55.36 ms | 90.4% / 88.8% / 78.5% | 98.93% | 97.0% | 96.3% / 73.6% | 63.1% | [link](https://drive.google.com/drive/folders/1mbs_u8to4bPycUk2coUiXuXb-IduT5EW) | |
| A12 | SFD-320x320-3CH-MobileNetV2-mod2-CaffeSSD-Float32-Dense   | 90K | 8.2 MB | 34.92 ms | 89.8% / 87.7% / 75.4% | 98.73% | 97.08% | 95.4% / 72.5% | 61.9% | [link](https://drive.google.com/drive/folders/1b7Li9CMw-okHEC34XITpP5h8pLYfpgqR?usp=sharing) | [link](https://drive.google.com/file/d/1dGsvKrHK5wC69OiCWMSbYQ97YLDCTeoT/view?usp=sharing) |
| A13 | SFD-320x320-3CH-MobileNetV2-mod3-CaffeSSD-Float32-Dense   | 90K | 8.2 MB | -- ms | 88.7% / 86.3% / 72.4% | --% | --% | --% / --% | --% | -- | |
| A14 | SFD-640x640-3CH-MobileNetV2-mod4-CaffeSSD-Float32-Dense   | --K | -- MB | -- ms | --.-% / --.-% / --.-% | --% | --% | --% / --% | --% | -- | |
| A15 | SFD-320x320-3CH-MobileNetV2-mod5-CaffeSSD-from:ImageNet-Float32-Dense   | --K | 3.9 MB | -- ms | 89.4% / 87.8% / 77.2% | --% | --% | --% / --% | --% | -- | |
| A16 | SFD-320x320-3CH-MobileNetV2-mod5-CaffeSSD-from:COCO-Float32-Dense   | 90K | 3.9 MB | -- ms | 89.7% / 88.1% / 77.3% | --% | --% | --% / --% | --% | -- | |

### Naming Convention
The following naming convention is used for models:

+ SFD-{Input}-{Arch}-{CaffeVersion}-{Precision}-{Method}

e.g. SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense

__Input__
1. 640x640 RGB 3 Channel : 640x640-3CH
2. 640x640 Gray 1 Channel : 640x640-1CH
3. 320x320 RGB 3 Channel : 320x320-3CH
4. 320x320 Gray 1 Channel : 320x320-1CH

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

# MTCNN Models

__Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks__


|   Model   |  Size (MB)  | Speed (ms) | Wider Accuracy (E/M/H) | AFW Accuracy | Pascal Accuracy |FDDB Accuracy (D/C) |UFDD Accuracy | Download URL |
| ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| --------------| 
| MTCNN-Original | 33 MB   | .. ms | 84.8% / 82.5% / 59.8% | ..% | ..% | 95.05% / 71.37% | ..% | [link](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) |
| MTCNN-light | 6.9 MB   | .. ms | 81.3% / 78.3% / 46.8% | 97.68% | 93.95% | 91.42% / 60.06% | 42.1%| [link](https://github.com/ghimiredhikura/MTCNN-light-face-detection) |
| MTCNN-CPP | 5.54 MB   | .. ms | 83.4% / 81.0% / 54.0% | 97.77% | 94.18% | 91.79% / 68.57% | 42.1%| [link](https://github.com/ghimiredhikura/mtcnn-cpp) |
