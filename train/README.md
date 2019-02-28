# SFD Trained Models
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
| N | mobilenetv2-modd11-n     | 3CH 320x320 | 7.29 MB | 0.360 fps |
| M | mobilenetv2-modd11-m     | 3CH 320x320 | 4.42 MB | 0.620 fps |
| L | mobilenetv2-modd11-l     | 3CH 320x320 | 3.61 MB | 1.160 fps |
| R | mobilenetv2-modd11-r     | 3CH 320x320 | 2.03 MB | 2.610 fps |
| E | mobilenetv2-modd11-e     | 3CH 320x320 | 1.06 MB | 4.680 fps |
| F | mobilenetv2-modd11-f     | 3CH 320x320 | 0.91 MB | 6.950 fps |
| G | mobilenetv2-modd11-g     | 3CH 320x320 | 0.49 MB | 12.560 fps |
| I | mobilenetv2-modd11-i     | 3CH 320x320 | 0.34 MB | 16.500 fps |
| J | mobilenetv2-modd11-j     | 3CH 320x320 | 0.22 MB | 19.290 fps |
| P | mobilenetv2-modd11-p     | 3CH 320x320 | 0.16 MB | 21.960 fps |

### ImageNet

The models are trained using ImageNet dataset with-out pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy | Download |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| L | mobilenetv2-mod11-l     | --K | --% | |
| E | mobilenetv2-mod11-e     | --K | --% | |
| F | mobilenetv2-mod11-f     | --K | --% | |
| G | mobilenetv2-mod11-g     | --K | --% | |
| I | mobilenetv2-mod11-i     | --K | --% | |

### COCO

The models are trained using COCO dataset with ImageNet pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy | Download |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| L | mobilenetv2-mod11-l     | --K | --% | |
| E | mobilenetv2-mod11-e     | --K | --% | |
| F | mobilenetv2-mod11-f     | --K | --% | |
| G | mobilenetv2-mod11-g     | --K | --% | |
| I | mobilenetv2-mod11-i     | --K | --% | |

## Widerface

The models are trained using WiderFace dataset without pre-training using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy (E/M/H) | Download |
| ------------- | ------------- | ------------- | ------------- |-------------| 
| N | mobilenetv2-mod11-n     | 90K | 88.84% / 88.16% / 78.93% | |
| M | mobilenetv2-mod11-m     | 90K | 88.57% / 86.45% / 75.56% | |
| L | mobilenetv2-mod11-l     | 90K | 86.74% / 84.57% / 73.15% | |
| R | mobilenetv2-mod11-r     | 90K | 84.03% / 80.45% / 65.45% | |
| E | mobilenetv2-mod11-e     | 90K | 80.79% / 76.79% / 59.0% | |
| F | mobilenetv2-mod11-f     | 90K | 78.2% / 74.12% / 53.39% | |
| G | mobilenetv2-mod11-g     | 90K | 75.03% / 70.15% / 46.46% | |
| I | mobilenetv2-mod11-i     | 90K | 70.15% / 65.82% / 40.66% | |
| J | mobilenetv2-mod11-j     | 90K | 67.63% / 62.61% / 36.51% | |
| P | mobilenetv2-mod11-p     | 90K | 59.57% / 53.09% / 28.0% | |

### Widerface Augmented

The models are trained using data augmented WiderFace dataset with pre-training using ImageNet using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy (E/M/H) | Download | 
| ------------- | ------------- | ------------- | ------------- |-------------|
| L | mobilenetv2-mod11-l     | 90K | 88.44% / 86.23% / 75.16% | |
| E | mobilenetv2-mod11-e     | 90K | 82.03% / 78.35% / 61.21% | |
| F | mobilenetv2-mod11-f     | 90K | 80.57% / 75.87% / 56.1% | |
| G | mobilenetv2-mod11-g     | 90K | 69.03% / 65.83% / 43.13% | |
| I | mobilenetv2-mod11-i     | 90K | --% / --% / --% | |

## Pedestrian

The models are trained using Pedestrian dataset with pre-training using ImageNet using NVIDIA-Caffe FP16.

| ID | Model Name | Iterations | Accuracy (E/M/H) | Download | 
| ------------- | ------------- | ------------- | ------------- |-------------|
| L | mobilenetv2-mod11-l     | 90K | --% / --% / --% | |
| E | mobilenetv2-mod11-e     | 90K | --% / --% / --% | |
| F | mobilenetv2-mod11-f     | 90K | --% / --% / --% | |
| G | mobilenetv2-mod11-g     | 90K | --% / --% / --% | |
| I | mobilenetv2-mod11-i     | 90K | --% / --% / --% | |

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
