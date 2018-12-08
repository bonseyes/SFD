# S³FD: Single Shot Scale-invariant Face Detector

### Introduction

S³FD is a real-time face detector, which performs superiorly on various scales of faces with a single deep neural network, especially for small faces. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1708.05237).
This repository intends to reproduce the results of the aforementioned paper.

### Contents

1. [Preparation](#preparation)
2. [Evaluation](#evaluation)
3. [Training](#training)
3. [Future Enhancements](#enhancements)


### Preparation

Follow the instructions [here](./docs/SSD-install.md) to download and install the proper Caffe version.


### Evaluation

To test the pre-trained models, read the [Test Instructions](./docs/Test-Instructions.md)


### Training 

To train new models using SFD, read the [Train Instructions](./docs/Train-Instructions.md)

### Enhancements 

#### CFENet to enhance small object detection through feature fusion
- [CFENet Paper](https://arxiv.org/pdf/1806.09790.pdf)
- [CFENet Code](https://github.com/qijiezhao/CFENet/tree/working)

#### PyramidMobileNet to add greater spatial information to convolutions
- [PyramidMobileNet Paper](https://arxiv.org/pdf/1811.07083.pdf)

#### DSFD uses a Feature Enhance Module with two shot pass
- [DSFD Paper](https://arxiv.org/pdf/1810.10220.pdf)

#### SRN - Selective Refinement Network
- [SRN Paper](https://arxiv.org/abs/1809.02693)
