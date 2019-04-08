# SFD Object Detection Trained Models
This folder contains training scripts and trained models for various datasets for different modified MobileNetV2 architectures.

```
train
│   README.md  
│
└───mobilenetv2-mod11-[ID]
│   │   README.md
│   │   model_mobilenetv2_mod11_[ID].py
│   │
│   └───imagenet
│       │   *.caffemodel
│       │   *.prototxt
│       │   generate_imagenet.py
│       │   generate_model_fp16.py
│       │   start_training.sh
│       │   start_testing.sh
│       │   ...
│   └───coco
│       │   *.caffemodel
│       │   *.prototxt
│       │   generate_coco.py
│       │   generate_model_fp16.py
│       │   start_training.sh
│       │   start_testing.sh
│       │   ...
│   └───widerface
│       │   *.caffemodel
│       │   *.prototxt
│       │   generate_widerface.py
│       │   generate_model_fp16.py
│       │   start_training.sh
│       │   start_testing.sh
│       │   ...
│   └───widerfaceaugmented
│       │   *.caffemodel
│       │   *.prototxt
│       │   generate_widerface_aug.py
│       │   generate_model_fp16.py
│       │   start_training.sh
│       │   start_testing.sh
│       │   ...
│   └───pedestrian
│       │   *.caffemodel
│       │   *.prototxt
│       │   generate_pedestrian.py
│       │   generate_model_fp16.py
│       │   start_training.sh
│       │   start_testing.sh
│       │   ...
│   
└───...
```

## Reference Architectures

| ID | Model Name | Input | Size (MB)   | 1xARM A53 FPS |
| ------------- | ------------- | ------------- | ------------- |-------------|
| N | mobilenetv2-mod11-n     | 3CH 320x320 | 7.29 MB | 0.360 fps |
| L | mobilenetv2-mod11-l     | 3CH 320x320 | 3.61 MB | 1.160 fps |
| E | mobilenetv2-mod11-e     | 3CH 320x320 | 1.06 MB | 4.680 fps |
| F | mobilenetv2-mod11-f     | 3CH 320x320 | 0.91 MB | 6.950 fps |
| G | mobilenetv2-mod11-g     | 3CH 320x320 | 0.49 MB | 12.560 fps |
| I | mobilenetv2-mod11-i     | 3CH 320x320 | 0.34 MB | 16.500 fps |

### ImageNet

The models are trained using ImageNet dataset with-out pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy: Top-1/Top-5  | Download |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| N | mobilenetv2-mod11-n     | 675K | 48.58% / 74.194 | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_n/imagenet) |
| L | mobilenetv2-mod11-l     | 850K | 48.0% / 74.0% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_l/imagenet) |
| E | mobilenetv2-mod11-e     | 525K | 31.1% / 56.9% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_e/imagenet) |
| F | mobilenetv2-mod11-f     | 525K | 26.4% / 50.5% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_f/imagenet) |
| G | mobilenetv2-mod11-g     | 550K | 21.0% / 43.9% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_g/imagenet) |
| I | mobilenetv2-mod11-i     | 550K | 14.1% / 32.5% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_i/imagenet) |

### Multi-Object Detection 23 Classes : nuScenes

The models are trained using nuScenes dataset with ImageNet pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy | Download |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| N | mobilenetv2-mod11-n     | --K | --% | |
| L | mobilenetv2-mod11-l     | --K | --% | |
| E | mobilenetv2-mod11-e     | --K | --% | |
| F | mobilenetv2-mod11-f     | --K | --% | |
| G | mobilenetv2-mod11-g     | --K | --% | |
| I | mobilenetv2-mod11-i     | --K | --% | |

### Multi-Object Detection 81 Classes : COCO

The models are trained using COCO dataset with ImageNet pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy | Download |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| N | mobilenetv2-mod11-n     | 150K | 19.6% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_n/coco) |
| L | mobilenetv2-mod11-l     | 150K | 17.9% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_l/coco) |
| E | mobilenetv2-mod11-e     | --K | --% | |
| F | mobilenetv2-mod11-f     | 150K | 10.1% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_f/coco) |
| G | mobilenetv2-mod11-g     | 150K | 6.5% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_g/coco) |
| I | mobilenetv2-mod11-i     | --K | --% | |

## Single-Object Detection : Widerface

The models are trained using WiderFace dataset without pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy (E/M/H) | Download |
| ------------- | ------------- | ------------- | ------------- |-------------|
| N | mobilenetv2-mod11-n     | 90K | 88.84% / 88.16% / 78.93% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_n/widerface) |
| L | mobilenetv2-mod11-l     | 90K | 86.74% / 84.57% / 73.15% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_l/widerface) |
| E | mobilenetv2-mod11-e     | 90K | 80.79% / 76.79% / 59.0% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_e/widerface) |
| F | mobilenetv2-mod11-f     | 90K | 78.2% / 74.12% / 53.39% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_f/widerface)|
| G | mobilenetv2-mod11-g     | 90K | 75.03% / 70.15% / 46.46% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_g/widerface) |
| I | mobilenetv2-mod11-i     | 90K | 70.15% / 65.82% / 40.66% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_i/widerface) |

###  Single-Object Detection : Widerface Data Augmented

The models are trained using data augmented WiderFace dataset with pre-training using ImageNet using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy (E/M/H) | Download | 
| ------------- | ------------- | ------------- | ------------- |-------------|
| N | mobilenetv2-mod11-n     | 90K | --.--% / --.--% / --.--% | |
| L | mobilenetv2-mod11-l     | 90K | 88.44% / 86.23% / 75.16% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_l/widerfaceaugmented) |
| E | mobilenetv2-mod11-e     | 90K | 82.03% / 78.35% / 61.21% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_e/widerfaceaugmented) |
| F | mobilenetv2-mod11-f     | 90K | 80.57% / 75.87% / 56.1% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_f/widerfaceaugmented) |-
| G | mobilenetv2-mod11-g     | 90K | 69.03% / 65.83% / 43.13% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_g/widerfaceaugmented) |
| I | mobilenetv2-mod11-i     | 90K | 66.47% / 63.41% / 37.63% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_i/widerfaceaugmented) |

## Single-Object Detection : Pedestrian

The models are trained using Pedestrian dataset with pre-training using ImageNet using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy | Download | 
| ------------- | ------------- | ------------- | ------------- |-------------|
| N | mobilenetv2-mod11-n     | 90K | --% |  |
| L | mobilenetv2-mod11-l     | 90K |  46.6% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_l/pedestrian) |
| E | mobilenetv2-mod11-e     | 90K | --% | |
| F | mobilenetv2-mod11-f     | 90K | 35.6% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_f/pedestrian) |
| G | mobilenetv2-mod11-g     | 90K | 30.7% | [link](https://github.com/bonseyes/SFD/tree/master/train/mobilenetv2_mod11_g/pedestrian) |
| I | mobilenetv2-mod11-i     | 90K | --% | |

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

|   Model   |  Size (MB)  | Speed (ms) | Wider Accuracy (E/M/H) | AFW Accuracy | Pascal Accuracy |FDDB Accuracy (D/C) |UFDD Accuracy | Bench source URL |
| ------------- | ------------- |------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------|
| Brodmann17FaceDetector |armv7l (4.55 MB)| ..ms| 62.5% / 48.3% / 20.1% | ..% | ..% | ..% / ..% | ..% | [link](https://github.com/ghimiredhikura/Brodmann17FaceDetector) |
| libfacedetection | 2.58 MB| .. ms | 68.7% / 63.5% / 37.4% | ..% | ..% | ..% / ..% | ..%| [link](https://github.com/ghimiredhikura/libfacedetection-vs2015) |
| opencv_ssd_face_fp16 | 5.10 MB | ..ms| 62.5% / 40.2% / 17.7% | ..% | ..% | ..% / ..% | ..%| [link](https://github.com/ghimiredhikura/face_detection_evaluation/tree/master/fd_eval_vs17) |
