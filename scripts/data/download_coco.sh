#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "COCO")

wget http://images.cocodataset.org/zips/train2017.zip -O $output_dir/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O $output_dir/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip -O $output_dir/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $output_dir/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip -O $output_dir/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip -O $output_dir/image_info_test2017.zip


cd $output_dir
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip
unzip stuff_annottations_trainval2017.zip
unzip image_info_test2017.zip
