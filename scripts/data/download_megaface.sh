#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh 

dst_dir=$1
output_dir=$(create_folder "$dst_dir" "MegaFace")

# Loosely Cropped (Padded)

#Your access information:
#Username: ghmdeepak@gmail.com
#Password: w4)02dO2om

wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_0.tar.gz -O $output_dir/identities_0.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_1.tar.gz -O $output_dir/identities_1.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_2.tar.gz -O $output_dir/identities_1.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_2.tar.gz -O $output_dir/identities_3.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_4.tar.gz -O $output_dir/identities_4.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_5.tar.gz -O $output_dir/identities_5.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_6.tar.gz -O $output_dir/identities_6.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_7.tar.gz -O $output_dir/identities_7.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_8.tar.gz -O $output_dir/identities_8.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_9.tar.gz -O $output_dir/identities_9.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_10.tar.gz -O $output_dir/identities_10.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_11.tar.gz -O $output_dir/identities_11.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_12.tar.gz -O $output_dir/identities_12.tar.gz
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_13.tar.gz -O $output_dir/identities_13.tar.gz

# Metadata (Detection coordinates, full image URLs, landmarks)
wget --user=ghmdeepak@gmail.com --ask-password megaface.cs.washington.edu/dataset/download/content/identities_meta.tar.gz -O $output_dir/identities_meta.tar.gz

cd $output_dir

for i in {0..13}
do
    tar xzfv identities_$i.tar.gz
done

tar xzfv identities_meta.tar.gz