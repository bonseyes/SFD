# Pretraining of MobileNetv2-SSDLite with COCO

### Usage
1. Generate the trainval_lmdb and test_lmdb from COCO original images. Follow the instructions [here](https://github.com/aghagol/caffe-ssd/tree/master/data/coco). You _will_ have to edit some scripts.
    For this experiment we used COCO 2017, and we have not used the minival2014 subset.  You will need to generate a `labelmap.prototxt`. There is one already generated in [`models/labelmap.prototxt`](./models/labelmap.prototxt), but it is also available inside `$CAFFE_ROOT/data/coco`.
3. Modify the `.prototxt` files inside `models` to use the generated databases.
4. Start training using `train.sh`. Test with `test.sh`.

### Note (from the [original repo](https://github.com/chuanqi305/MobileNetv2-SSDLite))
There are some differences between Caffe and tensorflow implementation:
1. The padding method 'SAME' in tensorflow sometimes use the [0, 0, 1, 1] paddings, means that top=0, left=0, bottom=1, right=1 padding. In caffe, there is no parameters can be used to do that kind of padding.
2. MobileNet on Tensorflow use ReLU6 layer y = min(max(x, 0), 6), but caffe has no ReLU6 layer. Replace ReLU6 with ReLU cause a bit accuracy drop in ssd-mobilenetv2, but very large drop in ssdlite-mobilenetv2. There is a ReLU6 layer implementation in the fork of [ssd](https://github.com/chuanqi305/ssd).

### Resources
 - [Official documentation of MobileNetV2-SSDLite](https://github.com/chuanqi305/MobileNetv2-SSDLite)

### TODOs
- TODO: check if the weights provided correspond to tensorflow initialized weights or already have some pretrainig
- DONE: check how to create the LMDB databases. Try to improve this document with that info. Check https://github.com/aghagol/caffe-ssd/tree/master/data/coco
- TODO: once LMDB databases were created, check training. If that works, try to separate train from val during training.
- TODO: test trained model
