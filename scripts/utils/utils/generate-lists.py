# This script generate a list with image relative path and corresponding relative path of annotation text file.
# This format is needed to generate the lmdb database to train SSD
# Note: this script consider the main path to data_root_dir = /media/sdf .. the user should enter the name of the folder just following this directory for both annotations and images.
# check the shell script that run this code.

import numpy as np
import os
import math
import argparse
import glob
import scandir
from scandir import scandir # python 2.7
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default=None)
parser.add_argument('--annot_dir', default=None)
parser.add_argument('--out_directory', default=None)
parser.add_argument('--mode', default='train')
parser.add_argument('--seperator_images', default=None)
parser.add_argument('--seperator_annot', default=None)
parser.add_argument('--wider_face', action='store_true')

args = parser.parse_args()

TrainValFlag = args.mode
 
AnnotPath = args.annot_dir
ImagesPath = args.images_dir
print("add " + str(AnnotPath) + " to " + TrainValFlag)
# list generated need to be relative path with respect to data_root directory
images_path = ImagesPath[ImagesPath.find(args.seperator_images) : ]   # split the path from this directory
annot_path = AnnotPath[AnnotPath.find(args.seperator_annot) : ]       # split the path from this directory

if TrainValFlag == "val":
	OutputTxtFile = os.path.join(args.out_directory, "val.txt")
	Val_name_size = os.path.join(args.out_directory, "val_name_size.txt")
	f = open(OutputTxtFile, "a")   # append
	fts = open(Val_name_size, "a")  
elif TrainValFlag == "train":
	OutputTxtFile = os.path.join(args.out_directory, "train.txt")
	f = open(OutputTxtFile, "a")  # append
else:
	print("Error in input mode. Allowed modes are 'train' and 'val' ...")

#print(args.wider_face)
if args.wider_face:
	for DirName in os.listdir(AnnotPath):
		#print(DirName)
		annotations_path = os.path.join(AnnotPath, DirName)
		annotations_path = os.path.abspath(annotations_path)
                ImagesPath_new = os.path.join(ImagesPath, DirName)
                print(ImagesPath_new)
		ImagesPath_new = os.path.abspath(ImagesPath_new)

		images_path = ImagesPath_new[ImagesPath_new.find(args.seperator_images) : ]   # split the path from this directory
		annot_path = annotations_path[annotations_path.find(args.seperator_annot) : ]       # split the path from this directory

		for files in scandir(annotations_path):
			if files.is_file() and (files.name.endswith('.txt')):
				annot_relative_path = os.path.join(annot_path, files.name)
				images_relative_path = os.path.join(images_path, os.path.splitext(files.name)[0] + ".jpg")                      
				f.write(str(images_relative_path) + " " + str(annot_relative_path) + "\n" )
                        
			if TrainValFlag == "val":
				im = Image.open(os.path.join(ImagesPath_new, os.path.splitext(files.name)[0] + ".jpg" ) )
				width, height = im.size
				fts.write(os.path.splitext(files.name)[0] + " " + str(height) + " " + str(width) + "\n")

else:
	for files in scandir(AnnotPath):
		if files.is_file() and (files.name.endswith('.txt')):
			annot_relative_path = os.path.join(annot_path, files.name)
			images_relative_path = os.path.join(images_path, os.path.splitext(files.name)[0] + ".jpg") 			
			f.write(str(images_relative_path) + " " + str(annot_relative_path) + "\n" )
			
			if TrainValFlag == "val":
				im = Image.open(os.path.join(ImagesPath, os.path.splitext(files.name)[0] + ".jpg" ) )
				width, height = im.size
				fts.write(os.path.splitext(files.name)[0] + " " + str(height) + " " + str(width) + "\n")
f.close()
if TrainValFlag == "val":
	fts.close()

