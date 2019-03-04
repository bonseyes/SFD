import json
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys
import time
import argparse
from distutils.dir_util import mkpath
from skimage import io
import os

sys.path.append("../PyramidBox-Pytorch/")
import detection_fns
from detection_fns import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='ai_challenger_keypoint_test_a_20180103')
parser.add_argument('--out_dir', default='annotations')
#parser.add_argument('--faces_out_dir', default='images_out')
#parser.add_argument('--background_out_dir', default='images_out')
parser.add_argument('--image_dir', default='keypoint_test_a_images_20180103')
parser.add_argument('--json_file', default='keypoint_test_a_annotations_20180103.json')
parser.add_argument('--confidence', type=float, default=0.5)

args = parser.parse_args()

# set variable paths to images and json file
image_dir = os.path.join(args.data_dir, args.image_dir)

json_file = os.path.join(args.data_dir, args.json_file)
annos = json.load(open(json_file, 'r'))
print('annos', len(annos))

target_annotation_dir = os.path.join(args.out_dir)
mkpath(target_annotation_dir)

for idx, anno in enumerate(annos):
    # Print status.
    if (idx + 1) % 1000 == 0 or (idx + 1) == len(annos):
        print(str(idx + 1) + ' / ' + str(len(annos)) + "test")

    # read images
    image_path = image_dir + "/" + anno['image_id'] + '.jpg';
    image = cv2.imread(image_dir + "/" + anno['image_id'] + '.jpg', cv2.IMREAD_COLOR)
    print (image_path)
    out_file = target_annotation_dir + "/" + anno['image_id'] + '.txt'
    f = open(out_file, 'w')

    max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.4 # the max size of input image for caffe
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
    shrink = max_im_shrink if max_im_shrink < 1 else 1
        
    # start detection
    det0 = detect_face(image, shrink, args.confidence)  # origin test
    det1 = flip_test(image, shrink, args.confidence)    # flip test
    [det2, det3] = multi_scale_test(image, max_im_shrink, args.confidence) #min(2,1400/min(image.shape[0],image.shape[1])))  #multi-scale test
    det4 = multi_scale_test_pyramid(image, max_im_shrink, args.confidence)
    det = np.row_stack((det0, det1, det2, det3, det4))
    dets = bbox_vote(det)

    # read keypoints from json
    #Keypoints_list = []
    #keypoints_all = anno['keypoint_annotations']
    #print(keypoints_all)
    #for key, value in keypoints_all.iteritems():
        #print(value)
        #Keypoints_list.append(value[36:])  # last 6 elements contain head and neck info
        #print(Keypoints_list )
        #print(Keypoints_list[0][1])

    person_bbox_list = []
    person_bbox_all = anno['human_annotations']
    for key, bbox in person_bbox_all.iteritems():
        person_bbox_list.append(bbox)
    
    #count = 0
    height, width, channels = image.shape
    det_list = []
    for i in range(dets.shape[0]):
        xmin = int(dets[i][0])
        ymin = int(dets[i][1])
        xmax = int(dets[i][2])
        ymax = int(dets[i][3])
        score = dets[i][4]
        #det_list.append([xmin, ymin, xmax, ymax])
        if score >= args.confidence:
            face_bbox = np.atleast_2d( np.asarray([1, xmin, ymin, xmax, ymax]) ) # [CLASS = 1, DETS] # face
            np.savetxt(f, face_bbox, fmt=["%d",]*5 , delimiter=" ")
            #cv2.rectangle(image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0, 255, 0), 1)

        #if score > args.confidence:
        #padding = 25
        #xmin_c = max(0, xmin- padding)
        #ymin_c = max(0, ymin- padding)
        #xmax_c = min(width, xmax+ padding)
        #ymax_c = min(height, ymax+ padding)
        #det_list.append([xmin_c, ymin_c, xmax_c, ymax_c])
        #print("face_bound: ", xmin_c, ymin_c, xmax_c, ymax_c, score)
        
        #for idx, human in enumerate(Keypoints_list):
            #print("human: ", human)
            #if human[0] >= xmin_c and human[0] <= xmax_c and human[3] >= xmin_c and human[3] <= xmax_c and human[1] >= (ymin_c-20) and human[1] <= (ymax_c+20) and human[4] >= (ymin_c-20) and human[4] <= (ymax_c+20):

                #face_box = image[ymin:ymax, xmin:xmax]
                #face_bbox = np.atleast_2d( np.asarray([1, xmin, ymin, xmax, ymax]) ) # [CLASS = 1, DETS] # face
                #np.savetxt(f, face_bbox, fmt=["%d",]*5 , delimiter=" ")

                #face_crop = image[ymin_c:ymax_c, xmin_c:xmax_c]
                #face_resized = cv2.resize(face_crop, (224, 224))
                #cv2.imshow("img_crop", face_resized)
                #cv2.waitKey(0)
                #face_name = target_images_dir + "/" + anno['image_id'] + '_' + str(count) +'.jpg'
                #face_out = cv2.imwrite(face_name, face_resized)
                #count+=1
                #cv2.rectangle(image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0, 255, 0), 1)
                #cv2.circle(image,(human[0],human[1]), 3, (0,0,255), -1)
                #cv2.circle(image,(human[3],human[4]), 3, (0,0,255), -1)
                #del Keypoints_list[idx]
    
    for idx, person in enumerate(person_bbox_list):
        P_xmin = person[0]
        P_ymin = person[1]
        P_xmax = person[2]
        P_ymax = person[3]
        person_bbox = np.atleast_2d( np.asarray([2, P_xmin, P_ymin, P_xmax, P_ymax]) ) # [CLASS = 2, DETS] # person
        np.savetxt(f, person_bbox, fmt=["%d",]*5 , delimiter=" ")
        #cv2.rectangle(image, (int(P_xmin),int(P_ymin)), (int(P_xmax),int(P_ymax)), (0, 0, 255), 1)

    #cv2.imshow("img", image)
    #cv2.waitKey(0)
    
    #count_neg = 0
    #neg_boxes = generate_neg_samples(image, det_list)
    #for item in neg_boxes:
    #    crop = image[item[1]:item[3], item[0]:item[2]]
    #    resized = cv2.resize(crop, (224, 224))
    #    name = target_negative_images_dir + "/" + anno['image_id'] + '_' + str(count_neg) +'.jpg'
    #    out = cv2.imwrite(name, resized)
    #    count_neg += 1


    #image_annotation_path = target_annotation_dir + "/" + anno['image_id'] + '.txt';
    #print(image_annotation_path)
    #f = open(image_annotation_path, 'w')
    #write_to_txt(f, dets, image_path)


