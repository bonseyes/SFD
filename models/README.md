# Model Naming
This folder contains pre-trained models for SFD. The following naming convention is used for models:

SFD-{Input}-{Arch}-{CaffeVersion}-{Precision}-{Method}

e.g.

SFD-640x640-3CH-VGG-CaffeSSD-Float32-Dense
SFD-640x640-1CH-VGG-CaffeSSD-Float32-Dense

SFD-640x640-3CH-VGG-NVIDIACaffe0.17-Float32-Dense
SFD-640x640-1CH-VGG-NVIDIACaffe0.17-Float32-Dense

## Input
1. 640x640 RGB 3 Channel : 640x640-3CH
2. 640x640 Gray 1 Channel : 640x640-1CH

## Architectures
1. VGG
2. MobileNetV1
3. MobileNetV2

## Caffe Versions
1. Caffe SSD : CaffeSSD [Link]
2. NVIDIA Caffe 0.17 : NVIDIACaffe0.17 [Link]
3. TIDSP Jacinto Caffe 0.16 : TIDSPJacinto0.16 [Link]
4. Bonseyes Caffe 0.16 : BonseyesCaffe0.16 [Link]

## Precision
1. Float32
2. Float16

## Training Method
1. Dense
2. Sparse
3. Quantized

# Original Model
TODO Add Link to original model
