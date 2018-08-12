# Train Instructions for COCO dataset:

- SSD installation steps and notes can be found in the directory: docs/SSD-install.md, Make sure to include $CAFFE_ROOT/python to your PYTHONPATH

# Preparing training data
- Follow the intruction of SSD to prepare the data and create the lmdb of COCO dataset: https://github.com/weiliu89/caffe/tree/ssd/data/coco <br />
- Before satrting step 4, you should know that we use  COCO 2014 and you should also download test set for 2015 (test2015). Also, you should add the files instances_minival2014.json and instances_valminusminival.json to annotations folder. You can download them from here: https://github.com/cocodataset/cocoapi/issues/54
- Then run step 4 after modifying the files to update the pathes to COCO datset.  

# Train the SSD model
- Use this version of caffe: https://github.com/bonseyes/caffe-SSD/tree/MobNetV2-COCO which is adapted to train SSD-MobileNetV2 with COCO, also the scripts in data/coco/ are updated, you just need to change the path to coco dataset. 

- When training is started you may face some problems related to dimensions of the data layers. This is because coco contains some images that are not 3 channels. So you need to add the flag force_color: true to the training and testing prototxt files. This modification is already done in the provided code. 



