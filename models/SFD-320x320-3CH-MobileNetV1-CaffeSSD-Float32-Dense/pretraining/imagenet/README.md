## Pretrain MobileNetv1 with ILSVRC12 - classification task

1. Make sure you have downloaded [Imagenet 2012](http://www.image-net.org/challenges/LSVRC/2012/) challenge dataset. A script for that is already provided in this repository: `${SFD}/scripts/data/download_imagenet.sh`

2. Download some extra files to create LMDBS: `${CAFFE_ROOT}/data/ilsvrc12/get_ilsvrc_aux.sh`

3. Use the train-test splits in `${CAFFE_ROOT}/data/ilsvrc12/{train,text,val}.txt` to generate the LMDBS using `create_imagenet.sh` utility from Caffe. Make sure you edit the following variables in the file:
    
    ```
    EXAMPLE=/your/output/dir/lmdbs
    DATA=/Imagenet/split_files/dir/  # where train.txt and val.txt are
    TOOLS=build/tools

    TRAIN_DATA_ROOT=/path/of/imagenet/train/images
    VAL_DATA_ROOT=/path/of/imagenet/val/images

    RESIZE=true
    ```

    And then run the script:

    ```
    ${CAFFE_ROOT}/example/imagenet/create_imagenet.sh
    ```

4. Generate the `.prototxt` files using `${SFD}/scripts/train/Classifier-ImageNet/MobileNetV1-Dense/generate_model.sh`. Make sure you modify the `--lmdb` argument to match your newly created LMDBs.

5. Train the model with `./start_training.sh`

### Resources

- [On creating LMDB Imagenet dataset](https://github.com/BVLC/caffe/tree/master/examples/imagenet)
- [Caffe implementation of MobileNet](https://github.com/shicai/MobileNet-Caffe)
