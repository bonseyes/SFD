#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "VGG_FACE2")

wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/licence.txt -O $output_dir/licence.txt
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/Readme.txt -O $output_dir/Readme.txt

wget --user=nviso --password=nvisovgg zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_train.tar.gz -O $output_dir/vggface2_train.tar.gz
wget --user=nviso --password=nvisovgg zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz -O $output_dir/vggface2_test.tar.gz

wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/MD5 -O $output_dir/MD5

wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta.tar.gz -O $output_dir/meta.tar.gz

wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/bb_landmark.tar.gz -O $output_dir/bb_landmark.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/vggface2_caffe_model.tar.gz -O $output_dir/vggface2_caffe_model.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/vggface2_matconvnet_model.tar.gz -O $output_dir/vggface2_matconvnet_model.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/dev_kit.tar.gz -O $output_dir/dev_kit.tar.gz


cd $output_dir

tar -xzf vggface2_train.tar.gz
tar -xzf vggface2_test.tar.gz
tar -xvf meta.tar.gz
tar -xvf bb_landmark.tar.gz
tar -xvf vggface2_caffe_model.tar.gz
tar -xvf vggface2_matconvnet_model.tar.gz
tar -xvf dev_kit.tar.gz