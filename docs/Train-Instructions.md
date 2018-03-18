# Train Instructions:

- Setup of training machine on Amazon  <br />
-> Version of AMI (Deep Learning Base AMI (Ubuntu) Version 3.0 (ami-07c2a77e)  <br />
-> Version of Amazon Instance > p3.2xlarge  <br />

- SSD installation steps and notws can be found in the directory: docs/SSD-install.md, Make sure to include $CAFFE_ROOT/python to your PYTHONPATH

1. Follow the intruction of SSD to create the lmdb of WIDER FACE. [DONE]  <br />

-> To start with wider face dataset, first we need to download it. If you work on server, you will face problem to download the data from google drive directly to the server. Therefore, in the "scripts" folder in this repo, we use the scripts "download-data.sh" and "google-drive.py" to solve this problem. Just put these scripts in the directory: $HOME/data, then run the script like this: ./download-data.sh. the data will be downloaded and unziped.  <br />

-> Useful link: https://github.com/weiliu89/caffe/wiki/Train-SSD-on-custom-dataset <br />
- To prepare the data and genearte lmdb database, next steps are done: <br />
 1- prepared script to modify annotations as required for lmdb and put them in seperate files: The annotations should be in the format: "label_id xmin ymin xmax ymax" and exist inside text files with the same names as their corresponding images, these files contain all bounding boxes in the annotations related to this image. This script exists in the scripts/ folder with the name "generate-train-annotations.py". We need to run this script twice, one for training and one for validation set, in each time just uncomment the corresponding lines in the script.  <br />

 2- prepared script to put each image with the corresponding annotation file, and list all of them in one text file. This script take the directories of the files generated from previous step, and the directory of the images, then it generate the list. The script exists in the scripts folder under the name "generate-lists.py". We need to run it twice as previous step, in each time uncomment the necessary lines. The output of this step should be in $SSD_ROOT/data/WIDER_FACE, which contans "train.txt", "val.txt", and "val_name_size.txt"  <br />

 3- modified the lmdb script "$SSD_ROOT/data/WIDER_FACE/create_data.sh" in SSD to make it accept txt annotations files not json or xml files. Then run this script which generated the lmdb database susccessfully in $HOME/data/WIDER_FACE  <br />

 4- prepared the file "labelmap_wider.prototxt" which contains the labels of the training and testing, 0 is background and 1 is face.  <br />

2. Modify the data augmentation code of SSD to make sure that it does not change the image ratio.
(Detail these steps here including which source files and what lines of code)  [DONE] <br />
-> [DONE] The augmentation strategy in the paper of SFD is followed, and the image ration is not changing. The details of this part is implemented in the python script for training.  <br />
-> In the file $CAFEE_ROOT/python/caffe/model_libs.py Change part of the function "VGGNetBody" in line 391 to be as follows "net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)" <br />


3. Modify the anchor match code of SSD to implement the 'scale compensation anchor matching strategy'.
(Detail these steps here including which source files and what lines of code)  [DONE] <br />

-> [DONE] modify the 'MatchBBox' function in bbox_util.cpp by adding extra stage to implement the stage 2 of anchor matching strategy. This part is added in the file bbox_util.cpp inside SSD code. The modified file will be attached with the final code. After this modification, caffe should be recombiled again. <br />

4. Train the model.
(Provide train script including amount of training time and validation loss curves and anything else useful) [DONE] <br />

-> [DONE] Prepare python script following SSD style that will include defining anchor box scales, stride, used VGG layers, augmentaion, then start training.  <br />
-> [DONE] Check training details from the paper (learning rate, max iterations, etc)  <br />
-> [DONE] Generate tarining loss and validation loss graphs after training is finished.  <br />

Training is done, with 120K iterations as noted in the SFD paper, an accuracy test with validation set is done each 5K iteartions. The scripts to plot training loss and test accuracy can be found in the path 'scripts/train/plot_train_graphs'. Plots of training loss vs. iterations, and validation accuracy vs. iterations are as follows:

![Alt text](assets/SFD_training_loss.png)
![Alt text](assets/SFD_validation_accuracy.png)

I think these numbers in validation accuracy is not normal!, but waiting for the results of test set. <br />
The model after training can be found here: https://drive.google.com/drive/folders/1WbTmDlUst-90lB8NC_KtcE11v49wW7W2?usp=sharing

# Run the training in the server

When the training script is run in the server, it will quit if we closed the terminal or if the internet is cut. To solve this we use the command "screen" as follows: <br />
After you ssh in, run screen. This will start another shell running within screen. Run your command, then do a "Ctrl-a-d". This will "disconnect" you from the screen session. At this point, you can log out or do anything else you'd like. When you want to re-connect to the screen session, just run "screen -RD" from the shell prompt.
more commands can be found here: http://aperiodic.net/screen/quick_reference




