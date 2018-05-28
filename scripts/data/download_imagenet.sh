#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "IMAGENET")

wget http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET.tar.gz -O $output_dir/ILSVRC2017_DET.tar.gz
wget http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET_test_new.tar.gz -O $output_dir/ILSVRC2017_DET_test_new.tar.gz

cd $output_dir
tar -xzf ILSVRC2017_DET.tar.gz
tar -xzf ILSVRC2017_DET_test_new.tar.gz
