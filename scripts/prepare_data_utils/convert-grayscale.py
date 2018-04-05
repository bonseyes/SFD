import cv2
import os
import scandir
from scandir import scandir

# This script generate grayscale images from the rgb images for the whole validation set

input_directory = "/home/ubuntu/data/WIDER_val/images"
output_directory = "/home/ubuntu/data/WIDER_val_grayscale/images"

for DirName in os.listdir(input_directory):
	in_path = os.path.join(input_directory, DirName)
	out_path = os.path.join(output_directory, DirName)
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	for image in scandir(in_path):
		img = os.path.join(in_path, image.name)
		gray_img = os.path.join(out_path, image.name)
		print img
		rgb = cv2.imread(img)
		gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(gray_img, gray)


