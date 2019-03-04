# This script reads the UFDD dataset validation annotations, and generate new files that can be used to train SSD based models.  
# The output should be in the form: "label_id xmin ymin xmax ymax"

import numpy as np
import os
import math
import argparse
import glob
import scandir
from scandir import scandir # python 2.7

# validation gt annotations
filename = "/media/sdf/UFDD/UFDD-annotationfile/UFDD_split/UFDD_val_bbx_gt.txt"
output_path = "/media/sdf/UFDD/annotations"

f = open(filename)
lines = f.read().splitlines()

for idx, l in enumerate(lines[0: ]): 
	values = l.split(".") 	# to detect the lines with image names  
	if values[-1] == 'jpg':
		#print(l)
		path_parts = l.split("/")
		#print(path_parts[0], path_parts[1])
		out_path = os.path.join(output_path, path_parts[0])
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		out_file = os.path.join(out_path, os.path.splitext(path_parts[1])[0] + ".txt")
		#f = open(out_file, 'w')
		#print(out_file)

		number_lines = int(lines[idx+1])
                if number_lines == 0:
                    continue

                f = open(out_file, 'w')
                print(out_file)

		results = []
		for bbox_line in lines[idx+2 : idx+2+number_lines]:
			#print bbox_line
			annot_values = bbox_line.split(" ")
			Xmin = int(annot_values[0])
			Ymin = int(annot_values[1])
			Xmax = int(annot_values[0]) + int(annot_values[2])
			Ymax = int(annot_values[1]) + int(annot_values[3])
			det_bbox = np.atleast_2d( np.asarray([1 , Xmin ,Ymin ,Xmax ,Ymax ]) ) # [CLASS = 1, DETS]
			np.savetxt(f, det_bbox, fmt=["%d",]*5 , delimiter=" ")
		f.close()

