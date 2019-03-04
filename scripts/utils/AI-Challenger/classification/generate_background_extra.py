import numpy as np
from random import randint
import argparse
import cv2
import os
from scandir import scandir

# This script generate extra negative samples from ImageNet validation set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default="/media/sdf/IMAGENET/ILSVRC12/train")
    parser.add_argument('--out_dir', default="/media/sdf/AI-Challenger/faces-NonFaces/background/train-set/extra_ImageNet_train")

    args = parser.parse_args()
    count = 0

    for folder in next(os.walk(args.images_dir))[1]: # return folder
        if count > 775000:
            break
        images_path = os.path.join(args.images_dir, folder)
        for files in scandir(images_path):
            if files.is_file() and files.name.endswith('.JPEG'):  # ImageNet is in *.JPEG format
                image_name = os.path.join(images_path, files.name)
                print("processing: ", image_name)
                img_base_name = os.path.splitext(files.name)[0]
                image = cv2.imread(image_name)
                box = generate_neg_samples(image)
                if box == None:
                    continue
                crop = image[box[1]:box[3], box[0]:box[2]]
                resized = cv2.resize(crop, (224, 224))
                name = args.out_dir + '/' + img_base_name + '.jpg'
                out = cv2.imwrite(name, resized)
                count+=1

def generate_neg_samples(image):
    height, width, channels = image.shape
    if height < 350 or width < 350:
        return None
    else:
        x = randint(1, width - 200)
        y = randint(1, height - 200)
        box_size = randint(160, 250)
        neg_sample = [x, y, min(x+box_size, width-1), min(y+box_size, height-1)]
        return neg_sample


if __name__ == "__main__":
    main()

