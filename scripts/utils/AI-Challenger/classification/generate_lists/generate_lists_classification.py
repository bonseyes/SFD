import numpy as np
from random import randint
import argparse
import cv2
import os
from scandir import scandir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default="/media/sdf/AI-Challenger/faces-NonFaces/validation-data")
    #parser.add_argument('--out_file', default="train.txt")

    args = parser.parse_args()

    for root, directories, filenames in os.walk(args.images_dir):
        #print(root, directories)
        for filename in filenames:
            image_path = os.path.join(root, filename) 
            #print(image_path.find("/pos/"))
            #print(image_path.find("/neg/"))
            if image_path.find("/pos/") != -1:
                label = 1
            elif image_path.find("/neg/") != -1:
                label = 0
            print(image_path + ' ' + str(label))


if __name__ == "__main__":
    main()

