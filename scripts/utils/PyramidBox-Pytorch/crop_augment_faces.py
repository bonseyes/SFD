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
# python2.7 face_detection_wider_format.py --data_dir=ai_challenger_keypoint_test_a_20180103 --out_dir=cropped --image_dir=keypoint_test_a_images_20180103 --json_file=keypoint_test_a_annotations_20180103.json --confidence=0.1

from __future__ import print_function
import os
import json
import time
import warnings
import argparse
import numpy as np
from distutils.dir_util import mkpath
from skimage import io

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio

from PIL import Image, ImageDraw
from pyramid import build_sfd
from layers import *
import cv2
import numpy as np
import math
from random import randint

os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.cuda.set_device(0)

print('Loading model..')
ssd_net = build_sfd('test', 640, 2)
net = ssd_net
net.load_state_dict(torch.load('./Res50_pyramid.pth'))
net.cuda()
net.eval()
print('Finished loading model!')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='ai_challenger_keypoint_test_a_20180103')
#parser.add_argument('--out_dir', default='annotations')
parser.add_argument('--faces_out_dir', default='images_out')
parser.add_argument('--background_out_dir', default='images_out')
parser.add_argument('--image_dir', default='keypoint_test_a_images_20180103')
parser.add_argument('--json_file', default='keypoint_test_a_annotations_20180103.json')
parser.add_argument('--confidence', type=float, default=0.1)
args = parser.parse_args()

# set variable paths to images and json file
image_dir = os.path.join(args.data_dir, args.image_dir)
images = os.listdir(image_dir)
print('images', len(images))

json_file = os.path.join(args.data_dir, args.json_file)
annos = json.load(open(json_file, 'r'))
print('annos', len(annos))

#target_annotation_dir = os.path.join(args.out_dir)
#mkpath(target_annotation_dir)
target_images_dir = args.faces_out_dir
mkpath(target_images_dir)
target_negative_images_dir = args.background_out_dir
mkpath(target_negative_images_dir)


start = time.time()
file_mapping = []

def detect_face(image, shrink):
    x = image

    shrink = 1.0
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    width = x.shape[1]
    height = x.shape[0]
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123],dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(), volatile=True)

    net.priorbox = PriorBoxLayer(width,height)
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    boxes=[]
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= args.confidence:
            score = detections[0,i,j,0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            boxes.append([pt[0],pt[1],pt[2],pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] / shrink
    det_ymin = boxes[:,1] / shrink
    det_xmax = boxes[:,2] / shrink
    det_ymax = boxes[:,3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det

def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image,0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image,1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def multi_scale_test_pyramid(image, max_shrink):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b

def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

def write_to_txt(f, det, image_path):
    f.write('{:s}\n'.format(image_path))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


def generate_neg_samples(image, det_list):
    height, width, channels = image.shape
    neg_samples_list = []

    for i in range( len(det_list) ):
        for det in det_list:
            try:
               x = randint(1, width - 150)
               y = randint(1, height - 150)
               box_size = randint(100, 220)
               if  (det[0] > x and det[0]< x+box_size) or (det[1] > y and det[1] < y+box_size) or (det[2] > x and det[2]< x+box_size) or (det[3] > y and det[3] < y+box_size) or ((det[2]-det[0])/2 > x and (det[2]-det[0])/2 < x+box_size) or ((det[3]-det[1])/2 > y and (det[3]-det[1])/2 < y+box_size):
                   continue
               else:
                   neg_samples_list.append([x, y, min(x+box_size, width-1), min(y+box_size, height-1)])
                   break
            
            except: 
                continue

    return neg_samples_list



for idx, anno in enumerate(annos):
    # Print status.
    if (idx + 1) % 1000 == 0 or (idx + 1) == len(annos):
        print(str(idx + 1) + ' / ' + str(len(annos)) + "test")

    # read images
    image_path = image_dir + "/" + anno['image_id'] + '.jpg';
    image = cv2.imread(image_dir + "/" + anno['image_id'] + '.jpg', cv2.IMREAD_COLOR)
    print (image_path)

    max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.4 # the max size of input image for caffe
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
    shrink = max_im_shrink if max_im_shrink < 1 else 1
        
    # start detection
    det0 = detect_face(image, shrink)  # origin test
    det1 = flip_test(image, shrink)    # flip test
    [det2, det3] = multi_scale_test(image, max_im_shrink) #min(2,1400/min(image.shape[0],image.shape[1])))  #multi-scale test
    det4 = multi_scale_test_pyramid(image, max_im_shrink)
    det = np.row_stack((det0, det1, det2, det3, det4))
    dets = bbox_vote(det)

    # read keypoints from json
    Keypoints_list = []
    keypoints_all = anno['keypoint_annotations']
    #print(keypoints_all)
    for key, value in keypoints_all.iteritems():
        #print(value)
        Keypoints_list.append(value[36:])  # last 6 elements contain head and neck info
        #print(Keypoints_list )
        #print(Keypoints_list[0][1])
    
    count = 0
    height, width, channels = image.shape
    det_list = []
    for i in range(dets.shape[0]):
        xmin = int(dets[i][0])
        ymin = int(dets[i][1])
        xmax = int(dets[i][2])
        ymax = int(dets[i][3])
        score = dets[i][4]
        #det_list.append([xmin, ymin, xmax, ymax])

        #if score > args.confidence:
        padding = 25
        xmin_c = max(0, xmin- padding)
        ymin_c = max(0, ymin- padding)
        xmax_c = min(width, xmax+ padding)
        ymax_c = min(height, ymax+ padding)
        det_list.append([xmin_c, ymin_c, xmax_c, ymax_c])
        print("face_bound: ", xmin_c, ymin_c, xmax_c, ymax_c, score)
        
        for idx, human in enumerate(Keypoints_list):
            #print("human: ", human)
            if human[0] >= xmin_c and human[0] <= xmax_c and human[3] >= xmin_c and human[3] <= xmax_c and human[1] >= (ymin_c-20) and human[1] <= (ymax_c+20) and human[4] >= (ymin_c-20) and human[4] <= (ymax_c+20):

                face_crop = image[ymin_c:ymax_c, xmin_c:xmax_c]
                face_resized = cv2.resize(face_crop, (224, 224))
                #cv2.imshow("img_crop", face_resized)
                #cv2.waitKey(0)
                face_name = target_images_dir + "/" + anno['image_id'] + '_' + str(count) +'.jpg'
                face_out = cv2.imwrite(face_name, face_resized)
                count+=1
                #cv2.rectangle(image, (int(xmin_c),int(ymin_c)), (int(xmax_c),int(ymax_c)), (0, 255, 0), 1)
                #cv2.circle(image,(human[0],human[1]), 3, (0,0,255), -1)
                #cv2.circle(image,(human[3],human[4]), 3, (0,0,255), -1)
                del Keypoints_list[idx]

    #cv2.imshow("img", image)
    #cv2.waitKey(0)
    
    count_neg = 0
    neg_boxes = generate_neg_samples(image, det_list)
    for item in neg_boxes:
        crop = image[item[1]:item[3], item[0]:item[2]]
        resized = cv2.resize(crop, (224, 224))
        name = target_negative_images_dir + "/" + anno['image_id'] + '_' + str(count_neg) +'.jpg'
        out = cv2.imwrite(name, resized)
        count_neg += 1


    #image_annotation_path = target_annotation_dir + "/" + anno['image_id'] + '.txt';
    #print(image_annotation_path)
    #f = open(image_annotation_path, 'w')
    #write_to_txt(f, dets, image_path)

print('Successfully processed all the images in %.2f seconds.' % (time.time() - start))
