import os
import json
import numpy as np
import argparse
from scandir import scandir
import cv2
import sys

sys.path.append("../PyramidBox-Pytorch/")
import detection_fns
from detection_fns import *

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default='/media/sdf/COCO/2014/images/val2014')
parser.add_argument('--json_dir', default='/media/sdf/COCO/2014/Annotations/val2014')
parser.add_argument('--out_dir', default='/media/sdf/COCO/2014/Annotations_person')
#parser.add_argument('--mode', default='faces_and_person')
parser.add_argument('--confidence', type=float, default=0.5)

args = parser.parse_args()

#images_input_folder = "/media/sdf/COCO/2014/images/val2014"
#json_input_folder = "/media/sdf/COCO/2014/Annotations/val2014"
#output_folder = "/media/sdf/COCO/2014/Annotations_person"

for files in scandir(args.json_dir):
    person_exists = False
    if files.is_file() and files.name.endswith('.json'):
        json_file = os.path.join(args.json_dir, files.name)
        print(json_file)
        out_file = args.out_dir + "/" + os.path.splitext(files.name)[0] + ".txt"
        #print(out_file)
        image_path = args.images_dir + "/" + os.path.splitext(files.name)[0] + ".jpg";
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #f = open(out_file, 'w')
        annos = json.load(open(json_file, 'r'))
        anno = annos["annotation"]
        if not anno:
            continue
        #print(len(anno))
        for obj in anno:
            category_id = obj["category_id"]
            is_crowd = obj["iscrowd"]
            if category_id == 1 and is_crowd != 1:
                #print(json_file)
                person_exists = True
                f = open(out_file, 'a')
                bbox = obj["bbox"]
                #print(bbox)
                P_xmin = bbox[0]
                P_ymin = bbox[1] 
                P_xmax = bbox[2] + bbox[0]
                P_ymax = bbox[3] + bbox[1] 
                person_bbox = np.atleast_2d( np.asarray([2, P_xmin, P_ymin, P_xmax, P_ymax]) ) # [CLASS = 2, DETS]
                np.savetxt(f, person_bbox, fmt=["%d",]*5 , delimiter=" ")
                #cv2.rectangle(image, (int(P_xmin),int(P_ymin)), (int(P_xmax),int(P_ymax)), (0, 255, 0), 1)
                f.close()
    
        if person_exists == True:
            # face det
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

            #det_list = []
            #print(dets)
            #if (dets.any() ):
            #f = open(out_file, 'a')
            for i in range(dets.shape[0]):
                xmin = int(dets[i][0])
                ymin = int(dets[i][1])
                xmax = int(dets[i][2])
                ymax = int(dets[i][3])
                score = dets[i][4]
                #if args.mode == "faces_only" or  args.mode == "faces_and_person":
                #f = open(out_file, 'a')
                if score >= args.confidence:
                    f = open(out_file, 'a')
                    face_bbox = np.atleast_2d(np.asarray([1, xmin, ymin, xmax, ymax]) ) # [CLASS = 1, DETS] # face
                    np.savetxt(f, face_bbox, fmt=["%d",]*5 , delimiter=" ")
                    #cv2.rectangle(image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0, 0, 255), 1)
                    f.close()

            #cv2.imshow("img", image)
            #cv2.waitKey(0)

