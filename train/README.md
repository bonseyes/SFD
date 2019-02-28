# SFD Trained Models
This folder contains training scripts and trained models for various datasets for different modified MobileNetV2 architectures.

```
train
│   README.md  
│
└───MobileNetV2-mod11-l
│   │   README.md
│   │
│   └───ImageNet
│       │   *.caffemodel
│       │   *.prototxt
│       │   *.sh
│       │   ...
│   └───COCO
│       │   *.caffemodel
│       │   *.prototxt
│       │   *.sh
│       │   ...
│   └───Widerface
│       │   *.caffemodel
│       │   *.prototxt
│       │   *.sh
│       │   ...
│   └───WiderfaceAugmented
│       │   *.caffemodel
│       │   *.prototxt
│       │   *.sh
│       │   ...
│   └───Pedestrian
│       │   *.caffemodel
│       │   *.prototxt
│       │   *.sh
│       │   ...
│   
└───MobileNetV2-modd11-X
    │   ...
```

## ImageNet

The models are trained using ImageNet dataset without pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Input | Iter | Size (MB)   | 1xARM A53 Speed | ImageNet (Top-5/ Top-1) | Download |
| ------------- | ------------- | ------------- | ------------- |-------------| --------------| -------------- | ----------------|
| L | MobileNetV2-modd11-l     | 3CH 320x320 | 90K | 3.61 MB | 1.160 fps | --% / --% | |
| E | MobileNetV2-modd11-e     | 3CH 320x320 | 90K | 1.06 MB | 4.680 fps | --% / --% | |
| F | MobileNetV2-modd11-f     | 3CH 320x320 | 90K | 0.91 MB | 6.950 fps | --% / --% | |
| G | MobileNetV2-modd11-g     | 3CH 320x320 | 90K | 0.49 MB | 12.560 fps | --% / --% | |
| I | MobileNetV2-modd11-i     | 3CH 320x320 | 90K | 0.34 MB | 16.500 fps | --% / --% | |

### COCO

The models are trained using COCO dataset with ImageNet pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Input | Iter | Size (MB)   | 1xARM A53 Speed | MAP | Download |
| ------------- | ------------- | ------------- | ------------- |-------------| --------------| -------------- | ----------------|
| L | MobileNetV2-modd11-l     | 3CH 320x320 | 90K | 3.61 MB | 1.160 fps | --% | |
| E | MobileNetV2-modd11-e     | 3CH 320x320 | 90K | 1.06 MB | 4.680 fps | --% | |
| F | MobileNetV2-modd11-f     | 3CH 320x320 | 90K | 0.91 MB | 6.950 fps | --% | |
| G | MobileNetV2-modd11-g     | 3CH 320x320 | 90K | 0.49 MB | 12.560 fps | --% | |
| I | MobileNetV2-modd11-i     | 3CH 320x320 | 90K | 0.34 MB | 16.500 fps | --% | |

## Widerface

The models are trained using WiderFace dataset without pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Input | Iter | Size (MB)   | 1xARM A53 Speed | Wider Accuracy (E/M/H) | Download |
| ------------- | ------------- | ------------- | ------------- |-------------|-------------| --------------| --------------| 
| N | MobileNetV2-modd11-n     | 3CH 320x320 | 90K | 7.29 MB | 0.360 fps | 88.84% / 88.16% / 78.93% | |
| M | MobileNetV2-modd11-m     | 3CH 320x320 | 90K | 4.42 MB | 0.620 fps | 88.57% / 86.45% / 75.56% | |
| L | MobileNetV2-modd11-l     | 3CH 320x320 | 90K | 3.61 MB | 1.160 fps | 86.74% / 84.57% / 73.15% | |
| R | MobileNetV2-modd11-r     | 3CH 320x320 | 90K | 2.03 MB | 2.610 fps | 84.03% / 80.45% / 65.45% | |
| E | MobileNetV2-modd11-e     | 3CH 320x320 | 90K | 1.06 MB | 4.680 fps | 80.79% / 76.79% / 59.0% | |
| F | MobileNetV2-modd11-f     | 3CH 320x320 | 90K | 0.91 MB | 6.950 fps | 78.2% / 74.12% / 53.39% | |
| G | MobileNetV2-modd11-g     | 3CH 320x320 | 90K | 0.49 MB | 12.560 fps | 75.03% / 70.15% / 46.46% | |
| I | MobileNetV2-modd11-i     | 3CH 320x320 | 90K | 0.34 MB | 16.500 fps | 70.15% / 65.82% / 40.66% | |
| J | MobileNetV2-modd11-j     | 3CH 320x320 | 90K | 0.22 MB | 19.290 fps | 67.63% / 62.61% / 36.51% | |
| P | MobileNetV2-modd11-p     | 3CH 320x320 | 90K | 0.16 MB | 21.960 fps | 59.57% / 53.09% / 28.0% | |

### Widerface Augmented

The models are trained using data augmented WiderFace dataset with pre-training using ImageNet using NVIDIA-Caffe FP16.

| ID | Model Name | Input | Iter | Size (MB)   | 1xARM A53 Speed | Wider Accuracy (E/M/H) | Download | 
| ------------- | ------------- | ------------- | ------------- |-------------|-------------| -------------- | ----------------|
| L | MobileNetV2-modd11-l     | 3CH 320x320 | 90K | 3.61 MB | 1.160 fps | 88.44% / 86.23% / 75.16% | |
| E | MobileNetV2-modd11-e     | 3CH 320x320 | 90K | 1.06 MB | 4.680 fps | 82.03% / 78.35% / 61.21% | |
| F | MobileNetV2-modd11-f     | 3CH 320x320 | 90K | 0.91 MB | 6.950 fps | 80.57% / 75.87% / 56.1% | |
| G | MobileNetV2-modd11-g     | 3CH 320x320 | 90K | 0.49 MB | 12.560 fps | 69.03% / 65.83% / 43.13% | |
| I | MobileNetV2-modd11-i     | 3CH 320x320 | 90K | 0.34 MB | 16.500 fps | --% / --% / --% | |

## Pedestrian

The models are trained using Pedestrian dataset with pre-training using ImageNet using NVIDIA-Caffe FP16.

| ID | Model Name | Input | Iter | Size (MB)   | 1xARM A53 Speed | Accuracy | Download | 
| ------------- | ------------- | ------------- | ------------- |-------------|-------------| -------------- | ----------------|
| L | MobileNetV2-modd11-l     | 3CH 320x320 | 90K | 3.61 MB | 1.160 fps | --% | |
| E | MobileNetV2-modd11-e     | 3CH 320x320 | 90K | 1.06 MB | 4.680 fps | --% | |
| F | MobileNetV2-modd11-f     | 3CH 320x320 | 90K | 0.91 MB | 6.950 fps | --% | |
| G | MobileNetV2-modd11-g     | 3CH 320x320 | 90K | 0.49 MB | 12.560 fps | --% | |
| I | MobileNetV2-modd11-i     | 3CH 320x320 | 90K | 0.34 MB | 16.500 fps | --% | |

### Naming Convention
The following naming convention is used for models:

+ SFD-{Input}-{Arch}-{CaffeVersion}-{Precision}-{Method}

e.g. SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense

__Input__
1. 640x640 RGB 3 Channel : 3CH 640x640
2. 640x640 Gray 1 Channel : 1CH 640x640
3. 320x320 RGB 3 Channel : 3CH 320x320
4. 320x320 Gray 1 Channel : 1CH 320x320

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

# Other - Very Fast/Light Models

**1. [https://github.com/brodmann17/Brodmann17FaceDetector](https://github.com/brodmann17/Brodmann17FaceDetector)**  
**2. [https://github.com/ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection)**  
**3. [https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)**  

|   Model   | Wider Accuracy (E/M/H) | AFW Accuracy | Pascal Accuracy |FDDB Accuracy (D/C) |UFDD Accuracy | Bench source URL |
| ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------|
| Brodmann17FaceDetector |  62.5% / 48.3% / 20.1% | ..% | ..% | ..% / ..% | ..% | [link](https://github.com/ghimiredhikura/Brodmann17FaceDetector) |
| libfacedetection | 68.7% / 63.5% / 37.4% | ..% | ..% | ..% / ..% | ..%| [link](https://github.com/ghimiredhikura/libfacedetection-vs2015) |
| opencv_ssd_face_fp16 | 62.5% / 40.2% / 17.7% | ..% | ..% | ..% / ..% | ..%| [link](https://github.com/ghimiredhikura/face_detection_evaluation/tree/master/fd_eval_vs17) |
