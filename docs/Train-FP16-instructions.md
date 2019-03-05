# Train Instructions:

- Setup of training machine on Amazon  <br />
-> Version of AMI (Deep Learning Base AMI (Ubuntu) Version 3.0 (ami-07c2a77e)  <br />
-> Version of Amazon Instance > p3.2xlarge  <br />

- To train with FP16 option, we need NVIDIA-Caffe, The version that has the changes of SFD can be found here:https://github.com/bonseyes/caffe-NVIDIA , don't forgert to checkout to the correct brance, now it is "SFD-0.17.1" <br />
- SSD installation steps and notes can be found in the directory: docs/SSD-install.md, Make sure to include $CAFFE_ROOT/python to your PYTHONPATH

# Preparing training data
1. Follow the intruction of SSD to create the lmdb of WIDER_FACE, you can find the scripts to do this step in the repo of caffe-NVIADA inside folder data/{dataset_name}  <br />

-> To start with wider face dataset, first we need to download it. If you work on server, you will face problem to download the data from google drive directly to the server. Therefore, in the "scripts" folder in this repo, we use the scripts "download-data.sh" and "google-drive.py" to solve this problem. Just put these scripts in the directory: $HOME/data, then run the script like this: ./download-data.sh. the data will be downloaded and unziped.  <br />

-> Useful link: https://github.com/weiliu89/caffe/wiki/Train-SSD-on-custom-dataset <br />

- To prepare the data and generate lmdb database, next steps are done: <br />
 1- prepare script to modify annotations as required for lmdb generation using caffe and put them in seperate files: The annotations should be in the format: "label_id xmin ymin xmax ymax" and exist inside text files with the same names as their corresponding images, these files contain all bounding boxes in the annotations related to this image. This script exists in the scripts/ folder with the name "generate-train-annotations.py". We need to run this script twice, one for training and one for validation set, in each time just uncomment the corresponding lines in the script.  <br />

 2- prepared script to put each image with the corresponding annotation file, and list all of them in one text file. This script take the directories of the files generated from previous step, and the directory of the images, then it generate the list. The script exists in the scripts folder under the name "generate-lists.py". We need to run it twice as previous step, in each time uncomment the necessary lines. The output of this step should be in $SSD_ROOT/data/WIDER_FACE, which contans "train.txt", "val.txt", and "val_name_size.txt"  <br />

 3- modify the lmdb script "$SSD_ROOT/data/WIDER_FACE/create_data.sh" in SSD to make it accept .txt annotations files not json or xml files. Then run this script which generate the lmdb database in $HOME/data/WIDER_FACE .. This script already exist in the repo of caffe-NVIDIA  <br />

 4- prepared the file "labelmap_wider.prototxt" which contains the labels of the training and testing, 0 is background and 1 is face.  <br />

# Modifications in Caffe codes

Note: These files and it modifications are already added to the repo caffe-NVIDIA <br />

1. Modify the data augmentation code of SSD to make sure that it does not change the image ratio. <br />
-> The augmentation strategy in the paper of SFD is followed, and the image ratio is not changing. The details of this part is implemented in the python script for training, it can also be found at the beginning of train.prototxt file.   <br />
-> modification in the caffe code "sampler.cpp" is done in the function "SampleBBox()", more info: https://github.com/sfzhang15/SFD/issues/20 and https://github.com/sfzhang15/SFD/issues/6 <br />

2. Modify the anchor match code of SSD to implement the 'scale compensation anchor matching strategy'. <br />
-> modify the 'MatchBBox' function in bbox_util.cpp by adding extra stage to implement the stage 2 of anchor matching strategy. This part is added in the file bbox_util.cpp inside SSD code. After this modification, caffe should be recombiled again. <br />

3. modify the function MeetEmitConstraint() in bbox_utils.cpp to ignore tiny faces < 6 pixels. Refer to this issue: https://github.com/sfzhang15/SFD/issues/7

4. Train the model.

-> Prepare python script following SSD style that will include defining anchor boxes scales, stride, used VGG layers, augmentaion, and the paths to the genearted logs. Then start training by runing the python script.  <br />
-> You first run the script generate_models.sh, which will generate the prototxt files, then run start_training.sh file. Plesae refer to the any model in the/models/MobileNetV2 folder inside thr caffe-NVIDIA repo <br />

# Run the training in the server

When the training script is run in the server, it will quit if we closed the terminal or if the internet is cut. To solve this we use the command "screen" as follows: <br />
After you ssh in, run screen. This will start another shell running within screen. Run your command, then do a "Ctrl-a-d". This will "disconnect" you from the screen session. At this point, you can log out or do anything else you'd like. When you want to re-connect to the screen session, just run "screen -RD" from the shell prompt.
more commands can be found here: http://aperiodic.net/screen/quick_reference
