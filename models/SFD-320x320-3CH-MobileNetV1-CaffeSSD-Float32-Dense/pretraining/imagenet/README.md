## Pretrain MobileNetv1 with ILSVRC12 - classification task

1. Make sure you have downloaded [Imagenet 2012](http://www.image-net.org/challenges/LSVRC/2012/) challenge dataset. A script for that is already provided in this repository: `${SFD}/scripts/data/download_imagenet.sh`. Note that the script will download train and validation sets.

2. Next is to preprocess the images to train the model. First, download some extra files using the following script: `${CAFFE_ROOT}/data/ilsvrc12/get_ilsvrc_aux.sh`. This will download, among other things, `{train,test,val}.txt` files with the images paths relatives to the Imagenet root folder. There are two options to preprocess data for Caffe:

   1. Using LMDBs (use Caffe `Data` layer as input): 

      Use the train-test splits in `${CAFFE_ROOT}/data/ilsvrc12/{train,text,val}.txt` to generate the LMDBS using `create_imagenet.sh` utility from Caffe. Make sure you edit the following variables in the file:
          
          ```
          EXAMPLE=/your/output/dir/lmdbs
          DATA=/Imagenet/split_files/dir/  # where train.txt and val.txt are (these files should have been downloaded by the get_ilsvrc_aux.sh script).
          TOOLS=build/tools
      
          TRAIN_DATA_ROOT=/path/of/imagenet/train/images
          VAL_DATA_ROOT=/path/of/imagenet/val/images
      
          RESIZE=true
          ```
      
          And then run the script:
      
          ```
          ${CAFFE_ROOT}/example/imagenet/create_imagenet.sh
          ```

   2. Using a plain text file (`train.txt`, `val.txt`, etc) with paths to images and their categories (use Caffe `ImageData` layer as input). The files should be the ones downloaded with `get_ilsvrc_aux.sh`. Make sure you resize the images manually before trainig the model and to properly shuffle `train.txt`. You can resize images by using the script in `${SFD_ROOT}/scripts/prepare_data_utils/resize-imagenet.py`

   We recommend using the second option since creating an Imagenet LMDB uses too much space.

4. Generate the `.prototxt` files using `${SFD}/scripts/train/Classifier-ImageNet/MobileNetV1-Dense/generate_model.sh`. 
   If you used LMDBs as input, make sure you modify the `--lmdb` argument to match your newly created LMDBs. Otherwise, make sure you change the input layer to `ImageData` and provide the path to the txt file containing all the images paths and labes.

5. Train the model with `./start_training.sh`. Make sure you have created proper folders to store logs and snapshots. Read the `start_training.sh` script to see how they should be named.

### Resources

- [On creating LMDB Imagenet dataset](https://github.com/BVLC/caffe/tree/master/examples/imagenet)
- [Caffe implementation of MobileNet](https://github.com/shicai/MobileNet-Caffe)


## Troubleshooting

- If Caffe does not recogize images, log, snapshots or prototxt files, it is probably because the paths are wrong. Try using absolute paths to avoid that issue.
