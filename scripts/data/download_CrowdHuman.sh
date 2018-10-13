#!/usr/bin/env bash

# Data set homepage: http://www.crowdhuman.org/

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "CrowdHuman")
drive=$root_dir/scripts/data/google-drive.py

python $drive 1XZm5L27eQasrvMq008l6vW8I8ysnavSM $output_dir/CrowdHuman_train01.zip
python $drive 1KgreKkhPfIiZHkl-x5K7p5LHKsfZss5q $output_dir/CrowdHuman_train02.zip
python $drive 1Af1rBvQSxOmXphoNNtKzAJyrEeeAF-aV $output_dir/CrowdHuman_train03.zip
python $drive 1dRRL6eKE1v_1Kb_R8nZGhTQ0HzYZ-Pss $output_dir/CrowdHuman_val.zip
python $drive 1uBB3psTLteVEP2Wg466DsYVx1eV8LacU $output_dir/annotation_train.odgt
python $drive 151-MHsdCni1izANEuZA3q3Pp7Jwpi-PX $output_dir/annotation_val.odgt

cd $output_dir
for file in "CrowdHuman_train01.zip" "CrowdHuman_train02.zip" "CrowdHuman_train03.zip" "CrowdHuman_val.zip"
do
    unzip $file.zip -d .
done
