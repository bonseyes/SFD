#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "NUSCENES")
drive=$root_dir/scripts/data/google-drive.py

python $drive 1pdXxiPxel0tg6keX7D_evELBXgDItRcL $output_dir/NUSCENES_MINI_DATASET_ROOT.zip
python $drive 1THfs0cE-WMxvdWoeB0FVNDP8J6S1s21Q $output_dir/NUSCENES_TEST_DATASET_ROOT.zip
python $drive 1S5wN2RH_qfOV-SQWzWg2qhczuZbo7mNO $output_dir/NUSCENES_TRAINVAL_DATASET_ROOT01.zip
python $drive 1KUK5WyDiEu0L4qf9xrUqX1AMrryd5aVX $output_dir/NUSCENES_TRAINVAL_DATASET_ROOT02.zip
python $drive 1RJIn9F0unXzqvWPS0n1VmbG3H_mCSRbr $output_dir/NUSCENES_TRAINVAL_DATASET_ROOT03.zip

cd $output_dir
for file in "NUSCENES_MINI_DATASET_ROOT" "UFDD-NUSCENES_TEST_DATASET_ROOT" \
            "NUSCENES_TRAINVAL_DATASET_ROOT01" "NUSCENES_TRAINVAL_DATASET_ROOT02" \
            "NUSCENES_TRAINVAL_DATASET_ROOT03"
do
    unzip $file.zip -d .
done

mv NUSCENES_TRAINVAL_DATASET_ROOT01 NUSCENES_TRAINVAL_DATASET_ROOT
mv NUSCENES_TRAINVAL_DATASET_ROOT02/samples/* NUSCENES_TRAINVAL_DATASET_ROOT/samples/
mv NUSCENES_TRAINVAL_DATASET_ROOT03/samples/* NUSCENES_TRAINVAL_DATASET_ROOT/samples/