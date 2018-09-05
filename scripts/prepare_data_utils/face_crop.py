# Copyright 2017 challenger.ai

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#############################################################

# this script is for cropping face images in Human Skeleton System Keypoints "https://challenger.ai/dataset/keypoint"
# usage example #
# python2.7 face_crop.py --data_dir=ai_challenger_keypoint_test_a_20180103 --out_dir=cropped --image_dir=keypoint_test_a_images_20180103 --json_file=keypoint_test_a_annotations_20180103.json

from __future__ import print_function
import os
import json
import time
import warnings
import argparse
import numpy as np
from distutils.dir_util import mkpath
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='ai_challenger_keypoint_test_a_20180103')
parser.add_argument('--out_dir', default='cropped')
parser.add_argument('--image_dir', default='keypoint_test_a_images_20180103')
parser.add_argument('--json_file', default='keypoint_test_a_annotations_20180103.json')
parser.add_argument('--ratio', type=float, default=2.)
parser.add_argument('--std', type=float, default=0.02)
args = parser.parse_args()

#image_dir = os.path.join(args.data_dir, 'keypoint_test_a_images_20180103/')
image_dir = os.path.join(args.data_dir, args.image_dir)
images = os.listdir(image_dir)
print('images', len(images))

#json_file = os.path.join(args.data_dir, 'keypoint_test_a_annotations_20180103.json')
json_file = os.path.join(args.data_dir, args.json_file)
annos = json.load(open(json_file, 'r'))
print('annos', len(annos))

target_image_dir = os.path.join(args.out_dir, 'face_images/')
mkpath(target_image_dir)

keypoints_std = np.ones(14) * args.std

start = time.time()
file_mapping = []

for idx, anno in enumerate(annos):

    # Print status.
    if (idx + 1) % 1000 == 0 or (idx + 1) == len(annos):
	print(str(idx + 1) + ' / ' + str(len(annos)))

    try:
        # Read image.
        img = io.imread(image_dir + "/" + anno['image_id'] + '.jpg')
        #print (image_dir + anno['image_id'] + '.jpg')

        height, width, channel = img.shape

        # For every human annotations.
        for key in anno['keypoint_annotations'].keys():

            # Read keypoint positions and the bounding box
            keypoints = np.reshape(anno['keypoint_annotations'][key], (14, 3))
        
	    bbox = anno['human_annotations'][key]
            mask = np.zeros((height, width), 'uint8')

            bbox_area = (bbox[3]-bbox[1]) * (bbox[2]-bbox[0])
            if bbox_area == 0:
                continue    
            dist_12_13 = np.linalg.norm(keypoints[12]-keypoints[13])
	    center = (keypoints[12]+keypoints[13])/2 	
	    
	    crop_left = center[0]-dist_12_13*0.75
	    crop_up = center[1]-dist_12_13*0.75
	    crop_right = center[0]+dist_12_13*0.75
	    crop_down = center[1]+dist_12_13*0.75

	    crop_left = np.max((int(crop_left), 0))
	    crop_up = np.max((int(crop_up), 0))
	    crop_right = np.min((int(crop_right), width-1))
	    crop_down = np.min((int(crop_down), height-1))
	
            # Crop the original image and the ground truth mask.
            img_crop = img[crop_up:crop_down, crop_left:crop_right, :]
	    
            # Suppress the warning of saving low contrast images.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Save the cropped images and cropped ground truth mask
                image_file = os.path.join(target_image_dir, anno['image_id'] + '_' + key + '.jpg')
                io.imsave(image_file, img_crop)
		file_mapping.append((image_file))
    except:
        continue

with open(os.path.join(args.out_dir, 'test.txt'), 'w') as f:
    for image_file in file_mapping:
        f.write('{}\n'.format(image_file))

print('Successfully processed all the images in %.2f seconds.' % (time.time() - start))
