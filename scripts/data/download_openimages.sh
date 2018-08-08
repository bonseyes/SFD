#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "OpenImages")
output_dir_1=$(create_folder "$output_dir" "Subset-with-Bounding-Boxes-600-classes")

# Subset with Bounding Boxes (600 classes) 
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/test_challenge.zip -O $output_dir_1/test_challenge.zip
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv -O $output_dir_1/train-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv -O $output_dir_1/validation-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv -O $output_dir_1/test-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels-boxable.csv -O $output_dir_1/train-annotations-human-imagelabels-boxable.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels-boxable.csv -O $output_dir_1/validation-annotations-human-imagelabels-boxable.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels-boxable.csv -O $output_dir_1/test-annotations-human-imagelabels-boxable.csv 
wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv -O $output_dir_1/train-images-boxable-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv -O $output_dir_1/validation-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv -O $output_dir_1/test-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv -O $output_dir_1/class-descriptions-boxable.csv 

output_dir_2=$(create_folder "$output_dir" "Subset-with-Image-Level-Labels-19,995-classes")

# install AWS command line interface 
sudo pip install awscli

# Subset with Image-Level Labels (19,995 classes)
aws s3 --no-sign-request cp s3://open-images-dataset/tar/challenge2018.tar.gz $output_dir_2/challenge2018.tar.gz
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels.csv -O $output_dir_2/train-annotations-human-imagelabels.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels.csv -O $output_dir_2/validation-annotations-human-imagelabels.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels.csv -O $output_dir_2/test-annotations-human-imagelabels.csv
wget https://storage.googleapis.com/openimages/2018_04/train/train-images-with-labels-with-rotation.csv -O $output_dir_2/train-images-with-labels-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv -O $output_dir_2/validation-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv -O $output_dir_2/test-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/class-descriptions.csv -O $output_dir_2/class-descriptions.csv
wget https://storage.googleapis.com/openimages/2018_04/classes-trainable.txt -O $output_dir_2/classes-trainable.txt

cd $output_dir_1
unzip test_challenge.zip

cd ../../../
cd $output_dir_2
tar xzfv challenge2018.tar.gz 