import json
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys

sys.path.append("../PyramidBox-Pytorch/")
import detection_fns
from detection_fns import *


parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default='/media/sdf/CrowdHuman/train/Images')
parser.add_argument('--odgt_file', default='/media/sdf/CrowdHuman/annotation_train.odgt')
parser.add_argument('--out_directory', default='/media/sdf/CrowdHuman/train/annotations')
parser.add_argument('--mode', default='faces_and_person')
parser.add_argument('--confidence', type=float, default=0.65)

args = parser.parse_args()

print("start processing")
with open(args.odgt_file) as ff:
    for line in ff:
        j_content = json.loads(line)
        #print(j_content["ID"])
        # read images
        image_path = args.images_dir + "/" + j_content['ID'] + '.jpg';
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        if width > 2500 or height > 2500:
            print("image skipped")
            continue
        print (image_path)
        out_file = args.out_directory + "/" + j_content['ID'] + '.txt'
        f = open(out_file, 'w')

        # fbox: full body, vbox: visible body, hbox: head
        gt_boxes = j_content["gtboxes"]
        for instance in gt_boxes:
            person_vis = True
            extra = instance["extra"]
            if "ignore" in extra:
                continue
            if instance["tag"] == "person":
                fbox = instance["fbox"]
                P_xmin = fbox[0]
                P_ymin = fbox[1]
                P_xmax = fbox[2]+fbox[0]
                P_ymax = fbox[3]+fbox[1]

                #hbox = instance["hbox"]
                #H_xmin = hbox[0]
                #H_ymin = hbox[1]
                #H_xmax = hbox[2]+hbox[0]
                #H_ymax = hbox[3]+hbox[1]
                vbox = instance["vbox"]
                #head_ignore = instance["head_attr"]["ignore"]

                fullBodyBoxSize = fbox[2]*fbox[3]
                visibleBodySize = vbox[2]*vbox[3]
                # visbility at least 30% 
                if (float(visibleBodySize) / float(fullBodyBoxSize)) < 0.25:
                    person_vis = False
                P_xmin = fbox[0]
                P_ymin = fbox[1]
                P_xmax = fbox[2]+fbox[0]
                P_ymax = fbox[3]+fbox[1]

                #if person_vis:
                    #cv2.rectangle(image, (int(P_xmin),int(P_ymin)), (int(P_xmax),int(P_ymax)), (0, 255, 0), 1)

                #if head_ignore == 0:
                    # TODO: Use PyramidBox
                    #cv2.rectangle(image, (int(H_xmin),int(H_ymin)), (int(H_xmax),int(H_ymax)), (255, 0, 0), 1)

                if args.mode == "person_only": # id = 1
                    if person_vis:
                        person_bbox = np.atleast_2d( np.asarray([1, P_xmin, P_ymin, P_xmax, P_ymax]) ) # [CLASS = 1, DETS]
                        np.savetxt(f, person_bbox, fmt=["%d",]*5 , delimiter=" ")
                elif args.mode == "faces_only": # id = 1
                    continue
                    #faces_bbox = np.atleast_2d( np.asarray([1 , H_xmin, H_ymin, H_xmax, H_ymax]) ) # [CLASS = 1, DETS]
                    #np.savetxt(f, faces_bbox, fmt=["%d",]*5 , delimiter=" ")
                elif args.mode == "faces_and_person": # id_face = 1, id_person = 2
                    #faces_bbox = np.atleast_2d( np.asarray([1 , H_xmin, H_ymin, H_xmax, H_ymax]) ) # [CLASS = 1, DETS]
                    #np.savetxt(f, faces_bbox, fmt=["%d",]*5 , delimiter=" ")
                    if person_vis:
                        person_bbox = np.atleast_2d( np.asarray([2, P_xmin, P_ymin, P_xmax, P_ymax]) ) # [CLASS = 2, DETS]
                        np.savetxt(f, person_bbox, fmt=["%d",]*5 , delimiter=" ")
        
        # face detection
        if args.mode == "faces_only" or  args.mode == "faces_and_person":
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

            det_list = []
            for i in range(dets.shape[0]):
                xmin = int(dets[i][0])
                ymin = int(dets[i][1])
                xmax = int(dets[i][2])
                ymax = int(dets[i][3])
                score = dets[i][4]
                #if args.mode == "faces_only" or  args.mode == "faces_and_person":
                if score >= args.confidence:
                    face_bbox = np.atleast_2d( np.asarray([1, xmin, ymin, xmax, ymax]) ) # [CLASS = 1, DETS] # face
                    np.savetxt(f, face_bbox, fmt=["%d",]*5 , delimiter=" ")
                    #cv2.rectangle(image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0, 0, 255), 1)
  
        #cv2.imshow("img", image)
        #cv2.waitKey(0)

