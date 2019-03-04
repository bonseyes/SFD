import numpy as np
import cv2
from pathlib import Path
import argparse
import sys

sys.path.append("../PyramidBox-Pytorch")
import detection_fns
from detection_fns import *

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default='/media/sdf/NIR-Database/images')
parser.add_argument('--out_dir', default='/media/sdf/NIR-Database/annotations')
parser.add_argument('--confidence', type=float, default=0.6)
parser.add_argument('--one_face', dest='one_face', action='store_true')
args = parser.parse_args()

one_face = args.one_face

images_dir = os.path.join(args.images_dir)
images = os.listdir(images_dir)
print('number of images ', len(images))

annotation_dir = os.path.join(args.out_dir)
mkpath(annotation_dir)

for img in images:
        image_path = args.images_dir + "/" + img  ;
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # face detection
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
        
        print(dets) 
        #out_file = args.out_dir + "/" + os.path.splitext(os.path.basename(img))[0] + '.txt'
        #f = open(out_file, 'w')

        det_list = []
        for i in range(dets.shape[0]):
            xmin = int(dets[i][0])
            ymin = int(dets[i][1])
            xmax = int(dets[i][2])
            ymax = int(dets[i][3])
            score = dets[i][4]
            
            if score >= args.confidence:
                out_file = args.out_dir + "/" + os.path.splitext(os.path.basename(img))[0] + '.txt'
                f = open(out_file, 'a')  # modify this according to your project

                face_bbox = np.atleast_2d( np.asarray([1, xmin, ymin, xmax, ymax]) ) # [CLASS = 1, DETS] # face
                np.savetxt(f, face_bbox, fmt=["%d",]*5 , delimiter=" ")
                #cv2.rectangle(image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0, 0, 255), 1)
                f.close()
                if one_face:
                    break
                #f.close()
            #cv2.imshow("img", image)
            #cv2.waitKey(0)

