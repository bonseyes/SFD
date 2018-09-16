#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "VGG_FACE2")

#id=nviso
#pass=nvisovgg

wget --user=nviso --password=nvisovgg 'zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_train.tar.gz' -O $output_dir/vggface2_train.tar.gz
wget --user=nviso --password=nvisovgg 'zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz' -O $output_dir/vggface2_test.tar.gz

wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/train_list.txt -O $output_dir/train_list.txt
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_list.txt -O $output_dir/test_list.txt
wget www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/identity_meta.csv -O $output_dir/identity_meta.csv
wget www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/class_overlap_vgg1_2.txt -O $output_dir/class_overlap_vgg1_2.txt
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_posetemp_imglist.txt -O $output_dir/test_posetemp_imglist.txt
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_agetemp_imglist.txt -O $output_dir/test_agetemp_imglist.txt
wget www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/bb_landmark.tar.gz -O $output_dir/bb_landmark.tar.gz

cd $output_dir

tar -xzf vggface2_train.tar.gz
tar -xzf vggface2_test.tar.gz
tar -xvf bb_landmark.tar.gz
