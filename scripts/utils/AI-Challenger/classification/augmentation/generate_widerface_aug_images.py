from __future__ import print_function, division
import argparse
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
import six.moves as sm
import re
import os
from scandir import scandir
from collections import defaultdict
import PIL.Image
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

import time
current_milli_time = lambda: int(round(time.time()))

np.random.seed(current_milli_time())
ia.seed(current_milli_time())


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default="/media/sdf/AI-Challenger/ai_challenger_keypoint_test_a_20180103/out/images/faces")
    parser.add_argument('--num_samples', default=5)
    parser.add_argument('--out_dir', default=None)


    args = parser.parse_args()

    #filename = "/media/sdf/AI-Challenger/ai_challenger_keypoint_test_a_20180103/out/images/faces/f5d05af3f88828abcafdcb80d428d48d5512c315_0.jpg"
    #generate_samples = 10
    input_folder = args.images_dir
    #out_dir = "/media/sdf/AI-Challenger/ai_challenger_keypoint_test_a_20180103/out/images/aug"
    out_dir = args.out_dir # output images in the same input folder
    for files in scandir(input_folder):
        if files.is_file() and files.name.endswith('.jpg'):
            image_name = os.path.join(input_folder, files.name)
            print("processing: ", image_name)
            img_base_name = os.path.splitext(files.name)[0]
            augment_sequential_images(image_name, args.num_samples, img_base_name, out_dir)


def create_sequence_imgaug():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            #iaa.Scale({"height": 320, "width": 568 }, interpolation=["linear", "cubic"]), #
            #iaa.Crop(px=((0), (124), (0), (124)), keep_size=False), #

            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            sometimes(iaa.Grayscale(alpha=(0.0, 1.0))), # Remove color information

            sometimes(iaa.Affine(
                scale=(0.8, 1.4), # scale images to 80-140% of their size, individually per axis
                translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)}, # translate by -20 to +20 percent (per axis)
                rotate=(-15, 15), # rotate by -15 to +15 degrees
                shear=(-10, 10), # shear by -10 to +10 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode="reflect" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 3),
                [
                    iaa.Multiply((0.2, 2.0)),
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.5)), # blur images with a sigma between 0 and 2.0
                        iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 5
                        iaa.MedianBlur(k=(3, 7)), # blur image using local medians with kernel sizes between 3 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 0.7), lightness=(0.8, 1.2)), # sharpen images
                    #iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.3)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    #iaa.SimplexNoiseAlpha(iaa.OneOf([
                    #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    #])),
                    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.10), size_percent=(0.02, 0.05), per_channel=0.15),
                    ]),
                    #iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.2), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    #iaa.OneOf([
                    #    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    #    iaa.FrequencyNoiseAlpha(
                    #        exponent=(-4, 0),
                    #        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                    #        second=iaa.ContrastNormalization((0.5, 2.0))
                    #    )
                    #]),
                    iaa.ContrastNormalization((0.6, 1.2), per_channel=0.5), # improve or worsen the contrast
                    #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    #sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=False
    )

    return seq

def augment_sequential_images(filename, generate_samples, img_base_name, out_dir):

    # Random seed
    ia.seed(current_milli_time())

    image = ndimage.imread(filename)
    seq = create_sequence_imgaug()

    # Run augmentation for N times
    for i in range(generate_samples):
        aug_det = seq.to_deterministic()
        newimage = aug_det.augment_image(image)
        newfilename = img_base_name + "_aug_" + str(i) + ".jpg" #"examples_newimage_%d.jpg" % i
        out_img = os.path.join(out_dir, newfilename)
        misc.imsave(out_img, newimage)


if __name__ == "__main__":
    main()
