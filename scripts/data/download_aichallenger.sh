#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "AIChallenger")

# Human Skeletal System Keypoints

#Your access information:
#Username: ghmdeepak@gmail.com
#Password: nvisoai

wget --user=ghmdeepak@gmail.com --ask-password http://ai-challenger.ufile.ucloud.com.cn/ai_challenger_keypoint_train_20170909.zip -O $output_dir/ai_challenger_keypoint_train_20170909.zip
wget --user=ghmdeepak@gmail.com --ask-password static1.challenger.ai/ai_challenger_keypoint_validation_20170911.zip -O $output_dir/ai_challenger_keypoint_validation_20170911.zip
wget --user=ghmdeepak@gmail.com --ask-password static1.challenger.ai/ai_challenger_keypoint_test_a_20180103.zip -O $output_dir/ai_challenger_keypoint_test_a_20180103.zip
wget --user=ghmdeepak@gmail.com --ask-password static1.challenger.ai/ai_challenger_keypoint_test_b_20180103.zip -O $output_dir/ai_challenger_keypoint_test_b_20180103.zip

cd $output_dir

unzip ai_challenger_keypoint_train_20170909.zip
unzip ai_challenger_keypoint_validation_20170911.zip
unzip ai_challenger_keypoint_test_a_20180103.zip
unzip ai_challenger_keypoint_test_b_20180103.zip
