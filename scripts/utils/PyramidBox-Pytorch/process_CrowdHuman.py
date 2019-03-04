import json
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys

#sys.path.append("../PyramidBox-Pytorch")
import detection_fns

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default='/media/sdf/CrowdHuman/train/Images')
parser.add_argument('--odgt_file', default='/media/sdf/CrowdHuman/annotation_train.odgt')
parser.add_argument('--out_directory', default='/media/sdf/CrowdHuman/train/annotations')
parser.add_argument('--mode', default='person_only')

args = parser.parse_args()

print("start processing")
with open(args.odgt_file) as f:
    for line in f:
        j_content = json.loads(line)
        #print(j_content["ID"])
        # read images
        image_path = args.images_dir + "/" + j_content['ID'] + '.jpg';
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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

                hbox = instance["hbox"]
                H_xmin = hbox[0]
                H_ymin = hbox[1]
                H_xmax = hbox[2]+hbox[0]
                H_ymax = hbox[3]+hbox[1]
                vbox = instance["vbox"]
                head_ignore = instance["head_attr"]["ignore"]

                fullBodyBoxSize = fbox[2]*fbox[3]
                visibleBodySize = vbox[2]*vbox[3]
                # visbility at least 30% 
                if (float(visibleBodySize) / float(fullBodyBoxSize)) < 0.3:
                    person_vis = False
                P_xmin = fbox[0]
                P_ymin = fbox[1]
                P_xmax = fbox[2]+fbox[0]
                P_ymax = fbox[3]+fbox[1]

                if person_vis:
                    cv2.rectangle(image, (int(P_xmin),int(P_ymin)), (int(P_xmax),int(P_ymax)), (0, 255, 0), 1)

                if head_ignore == 0:
                    # TODO: Use PyramidBox
                    cv2.rectangle(image, (int(H_xmin),int(H_ymin)), (int(H_xmax),int(H_ymax)), (255, 0, 0), 1)


                if args.mode == "person_only": # id = 1
                    if person_vis:
                        person_bbox = np.atleast_2d( np.asarray([1, P_xmin, P_ymin, P_xmax, P_ymax]) ) # [CLASS = 1, DETS]
                        np.savetxt(f, person_bbox, fmt=["%d",]*5 , delimiter=" ")
                elif args.mode == "faces_only": # id = 1
                    faces_bbox = np.atleast_2d( np.asarray([1 , H_xmin, H_ymin, H_xmax, H_ymax]) ) # [CLASS = 1, DETS]
                    np.savetxt(f, faces_bbox, fmt=["%d",]*5 , delimiter=" ")
                elif args.mode == "faces_and_person": # id_face = 1, id_person = 2
                    faces_bbox = np.atleast_2d( np.asarray([1 , H_xmin, H_ymin, H_xmax, H_ymax]) ) # [CLASS = 1, DETS]
                    np.savetxt(f, faces_bbox, fmt=["%d",]*5 , delimiter=" ")
                    if person_vis:
                        person_bbox = np.atleast_2d( np.asarray([2, P_xmin, P_ymin, P_xmax, P_ymax]) ) # [CLASS = 2, DETS]
                        np.savetxt(f, person_bbox, fmt=["%d",]*5 , delimiter=" ")


        cv2.imshow("img", image)
        cv2.waitKey(0)

