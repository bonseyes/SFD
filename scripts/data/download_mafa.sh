#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "MAFA")
drive=$root_dir/scripts/data/google-drive.py


# download MAFA Testing Images
python $drive 1jJHdmmscqxvNQ2dxKUrLaHqW3w1Yo_9S $output_dir/test-images.zip
python $drive 1uN0a4P0wAFwJLid_r7VHFs0KUcizIRGN $output_dir/MAFA-Label-Test.zip

# waiting baidu account for MAFA Training Images and Face annotations

cd $output_dir
for file in "test-images" "MAFA-Label-Test"
do
    unzip $file.zip -d .
done