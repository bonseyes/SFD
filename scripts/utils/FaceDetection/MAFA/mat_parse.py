from scipy.io import loadmat
import argparse
import sys
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_mat', default='LabelTestAll.mat')
parser.add_argument('--out_directory', default='out_test')
parser.add_argument('--mode', default='train')
args = parser.parse_args()

if args.mode == 'train':
    annot_raw = loadmat(args.input_mat)['label_train']
    name_idx = 1
elif args.mode == 'test':
    annot_raw = loadmat(args.input_mat)['LabelTest']
    name_idx = 0

annot_all = annot_raw[0]
print(annot_all)

for row in annot_all:
    name = os.path.splitext(str(row[name_idx])[3:])[0]
    out_file = os.path.join(args.out_directory, name + ".txt")
    print(out_file)
    f = open(out_file, 'w')

    annot = row[name_idx+1] # annot for this image
    #print(annot)
    for face in annot:
        #print(face)
        #face_type = int(face[4])
        Xmin = int(face[0])
        Ymin = int(face[1])
        Xmax = int(face[0]) + int(face[2])
        Ymax = int(face[1]) + int(face[3])
        #if face_type != 3:
        det_bbox = np.atleast_2d( np.asarray([1 , Xmin ,Ymin ,Xmax ,Ymax ]) ) # [CLASS = 1, DETS]
        np.savetxt(f, det_bbox, fmt=["%d",]*5 , delimiter=" ")
    f.close()

