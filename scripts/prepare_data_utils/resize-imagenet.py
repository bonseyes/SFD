# Taken and adapted from FastAI: https://github.com/fastai/imagenet-fast/blob/master/scripts/resize.py
# It should work with both Python2.7 and Python3.*
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
import os
import os.path
import sys


cpus = multiprocessing.cpu_count()
cpus = min(48, cpus)

szs = (256, )  # (int(160*1.25),)
dst_folder_name = 'resized_256'


def resize_img(p, im, fn, sz):
    w, h = im.size
    ratio = min(h / sz, w / sz)
    im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)
    new_fn = os.path.join(p, '../..', 'resized_256', os.path.dirname(fn).split('/')[-2], os.path.dirname(fn).split('/')[-1])

    if not os.path.exists(new_fn):
        os.makedirs(new_fn)

    new_fn = os.path.join(new_fn, os.path.basename(fn))
    print(new_fn)
    im.save(new_fn)


def resizes(p, fn):
    im = Image.open(fn)
    for sz in szs:
        resize_img(p, im, fn, sz)


def resize_imgs(p):
    files = [os.path.join(dirpath, f) for dirpath, _, filenames in os.walk(p) for f in filenames if ".JPEG" in f]
    with ProcessPoolExecutor(cpus) as e:
        e.map(partial(resizes, p), files)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("You must provide the path where the original images are.")

    orig_imgs = sys.argv[1]

    new_path = os.path.join(orig_imgs, '..', dst_folder_name)

    # Create destination directories
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for ds in ('val','train'): 
        if not os.path.exists(new_path):
            os.makedirs(os.path.join(new_path, ds))

    for ds in ('val','train'):
        resize_imgs(os.path.join(orig_imgs, ds))
