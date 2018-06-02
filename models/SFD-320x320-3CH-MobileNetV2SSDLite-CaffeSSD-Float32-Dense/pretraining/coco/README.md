# Pretraining of MobileNetv2-SSDLite with COCO

### Usage
1. Generate the trainval_lmdb and test_lmdb from COCO original images. Follow the instructions [here](https://github.com/aghagol/caffe-ssd/tree/master/data/coco). You _will_ have to edit some scripts (paths, shapes, datasets names, etc).
    For this experiment we used COCO 2017, and we have not used the minival2014 subset.  You will need to generate a `labelmap.prototxt`. There is one already generated in [`models/labelmap.prototxt`](./models/labelmap.prototxt), but it is also available inside `$CAFFE_ROOT/data/coco`.
3. Modify the `.prototxt` files inside `models` to use the generated databases. These `.prototxt` have been taken from [here](https://github.com/chuanqi305/MobileNetv2-SSDLite/tree/master/ssdlite/coco). The `engine: CAFFE` line has been uncommented to make it work without CUDNN. If you are not using [this version of Caffe SSD](https://github.com/chuanqi305/ssd), then you will have to replace all the `ReLU6` layers by regular `ReLU` layers. There are some known issues of doing this, please refer to [Notes](#notes)
4. Start training using `train.sh`. Test with `test.sh`.

### Notes 
(Taken from the [original repo](https://github.com/chuanqi305/MobileNetv2-SSDLite))

There are some differences between Caffe and tensorflow implementation:
1. The padding method 'SAME' in tensorflow sometimes use the [0, 0, 1, 1] paddings, means that top=0, left=0, bottom=1, right=1 padding. In caffe, there is no parameters can be used to do that kind of padding.
2. MobileNet on Tensorflow use ReLU6 layer y = min(max(x, 0), 6), but caffe has no ReLU6 layer. Replace ReLU6 with ReLU cause a bit accuracy drop in ssd-mobilenetv2, but very large drop in ssdlite-mobilenetv2. There is a ReLU6 layer implementation in the fork of [ssd](https://github.com/chuanqi305/ssd).
3. When using [this implementation of Caffe](), if get the following error during training `[... math_functions.cpp:250] Check failed: a <= b (0 vs. -1.19209e-07)`, try [this solution](https://github.com/weiliu89/caffe/issues/669#issuecomment-339542120)

### Resources
 - [Official documentation of MobileNetV2-SSDLite](https://github.com/chuanqi305/MobileNetv2-SSDLite)

### TODOs
- DONE: check how to create the LMDB databases. Try to improve this document with that info. Check https://github.com/aghagol/caffe-ssd/tree/master/data/coco
- DONE: once LMDB databases were created, check training. TRAINING WORKS

- TODO: check if the weights provided correspond to tensorflow initialized weights or already have some pretrainig
- TODO: check that the dataset has been built correctly using the Ipython notebook of the authors.
- TODO: If everything works, try to separate train from val during training.
- TODO: test trained model
