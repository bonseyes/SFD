#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "UFDD")
drive=$root_dir/scripts/data/google-drive.py

python $drive 1bZGzDx_CgNnxoRdLnmMLiZ3F9k5lnY4J $output_dir/UFDD_information.zip
python $drive 1aGR7FryrRuS86S9LBAqFksy-QDqsgBRV $output_dir/UFDD-annotationfile.zip
python $drive 1o-lsXB7XLc4F39zQyZgwrabWyN1M5NBY $output_dir/UFDD_val.zip
python $drive 1jSa_4O6k5knELVKhBVdL9Ma_WQlxJ3ys $output_dir/README.txt

cd $output_dir
for file in "UFDD_information" "UFDD-annotationfile" "UFDD_val"
do
    unzip $file.zip -d .
done