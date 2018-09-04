# SFD Models
This folder contains pre-trained models for SFD using different architectures and training environments.

__Original Models__

| Model ID    | Model Name    | Input | Iterations   | Size (MB)   | Speed (ms)  | Wider Accuracy (E/M/H) | Download | Pretraining (Imagenet) |
| ------------- | ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| 
| A1 | Original                                        | 3CH 640x640 | 120K | 86 MB | 31.24 ms | 93.9% / 92.2% / 83.4% | [link](https://drive.google.com/file/d/1CboBIsjcDQ-FC1rMES6IjTl6sYQDoD6u/view) | [pretrained-model](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6) |
| A2 | SFD-VGG     | 3CH 640x640 | 120K | 86 MB | 30.28 ms | 93.4% / 92.0% / 82.3% | [link](https://drive.google.com/drive/u/0/folders/1WbTmDlUst-90lB8NC_KtcE11v49wW7W2) | [pretrained-model](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6) |

__MobileNetV1__

| Model ID    | Model Name    | Input | Iterations   | Size (MB)   | Speed (ms)  | Wider Accuracy (E/M/H) | Download | Pretraining (Imagenet) |
| ------------- | ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| 
| A7 | SFD-MobileNetV1      | 3CH 320x320 | 100K | 20 MB | 36.80 ms | 86.9% / 82.5% / 65.5% | [link](https://drive.google.com/open?id=1_tmrTB0HYzSq3gFPKUWlfKtHQn6WIWNM) | [pretrained-model](https://drive.google.com/file/d/0B3gersZ2cHIxVFI1Rjd5aDgwOG8/view?usp=sharing) |

__MobileNetV2__

| Model ID    | Model Name    | Input | Iterations   | Size (MB)   | Speed (ms)  | Wider Accuracy (E/M/H) | Download | Pretraining (Imagenet) |
| ------------- | ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| 
| A11 | SFD-MobileNetV2-mod1  | 3CH 320x320 | -- | 14 MB | 55.36 ms | 90.4% / 88.8% / 78.5% | [link](https://drive.google.com/drive/folders/1mbs_u8to4bPycUk2coUiXuXb-IduT5EW) | [pretrained-model](https://drive.google.com/file/d/1ZFcV4MoIzJfUYlwVI47_738__fUte_9p/view?usp=sharing) |
| A11_2 | SFD-MobileNetV2-mod1  | 3CH 320x320 | -- | 12 MB | -- ms | 89.9% / 88.0% / 77.2% | | |
| A12 | SFD-MobileNetV2-mod2   | 3CH 320x320 | -- | 8.2 MB | -- ms | 89.8% / 87.7% / 75.4% |  [link](https://drive.google.com/drive/folders/1b7Li9CMw-okHEC34XITpP5h8pLYfpgqR?usp=sharing) | [pretrained-model](https://drive.google.com/open?id=1FTTlmqi9KOdoQFXOrMb5QnNF1SzKPbNN) |
| A13 | SFD-MobileNetV2-mod3   | 3CH 320x320 | -- | 10 MB | -- ms | 88.7% / 86.3% / 72.4% | -- | [pretrained-model](https://drive.google.com/file/d/1LV89HdBxctoOWP9ixabpipMKMS1Ftyrm/view?usp=sharing) |
| A14 | SFD-MobileNetV2-mod4   | 3CH 320x320 | -- | 10 MB | -- ms | -- | !! | [pretrained-model](https://drive.google.com/file/d/1ufB-tsZtlRDY7j4GcujKhxQM6UijxjR9/view?usp=sharing) |
| A15 | SFD-MobileNetV2-mod5   | 3CH 320x320 | 90K | 3.9 MB | -- ms | 89.4% / 87.8% / 77.2% | [link](https://drive.google.com/open?id=1LpwLW1NyN6uYw0q9kD0vBQ6Iy6T1_DLz) | [pretrained-model](https://drive.google.com/open?id=1C3eN-w-rgfvfkdt5dFRzRS-MO6GfXtBj) |
| A18 | SFD-MobileNetV2-mod5   | 3CH 640x640 | 90K | 3.9 MB | -- ms | 90.6% / 88.5% / 75.0% | [link](https://drive.google.com/open?id=1AWLwyTzxZ1TN_8qdKhDg4gwJge-9kJYG) | [pretrained-model](https://drive.google.com/open?id=1C3eN-w-rgfvfkdt5dFRzRS-MO6GfXtBj) |

__NVIDIA-Caffe FP16 Models__

| Model ID    | Model Name    | Input | Iterations   | Size (MB)   | Speed (ms)  | Wider Accuracy (E/M/H) | Download | Pretraining (Imagenet) |
| ------------- | ------------- | ------------- | ------------- |-------------|-------------| --------------| ----------------| --------------| 
| B1 | SFD-VGG-FP16     | 3CH 320x320 | 120K | 86 MB | -- ms | 90.0% / 89.0% / 79.5% | [link](https://drive.google.com/open?id=1B3N0MayY46jgoRzKFFPYYdsp9rLnKj_L) | NO pretrained model |
| B2 | SFD-MobileNetV2-mod5-FP16     | 3CH 320x320 | 90K | 3.9 MB | -- ms | 88.3% / 86.7% / 78.0% | [link](https://drive.google.com/open?id=1PA-LaOatq0T0LyTgRPJda43XtOay46mW) | NO pretrained model |
| B3 | SFD-MobileNetV2-mod5-FP16-pretrained    | 3CH 320x320 | 90K | 3.9 MB | -- ms | 89.2% / 87.9% / 77.7% | [link](https://drive.google.com/drive/folders/1nofs6b3OzVx_4jLTG9stjcgp1T2CoUYc?usp=sharing) | [pretrained model fp32](https://drive.google.com/open?id=1C3eN-w-rgfvfkdt5dFRzRS-MO6GfXtBj) |


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
