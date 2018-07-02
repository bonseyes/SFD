#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "IMAGENET")

wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar -O $output_dir/ILSVRC2012_img_train.tar
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar -O $output_dir/ILSVRC2012_img_val.tar

cd $output_dir
tar -xzf ILSVRC2012_img_train.tar
tar -xzf ILSVRC2012_img_val.tar
