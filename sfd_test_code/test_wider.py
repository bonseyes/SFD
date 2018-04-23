from sfd_detector import SFD_NET
import argparse
import caffe
import cv2
import numpy as np
import os
import os.path
import scipy.io as sio


def multi_scale_test(net, image, max_im_shrink):
    # Shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = net.detect(image, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # Enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = net.detect(image, bt)

    # Enlarge small image x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_ = net.detect(image, bt)
            det_b = np.row_stack((det_b, det_))
            bt *= 2
        det_ = net.detect(image, bt)
        det_b = np.row_stack((det_b, det_))

    # Enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = net.detect(image_f, shrink)

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
    dets = np.zeros((0, 5))
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

        # Get needed merge det and delete these det
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


def write_to_txt(f, det):
    f.write('{:s}\n'.format(event[0][0].encode('utf-8') + '/' + im_name + '.jpg'))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in xrange(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test code for a SFD trained model using WIDER FACE dataset.')
    parser.add_argument('-p', '--path', type=str, help='Dataset path', required=True)
    parser.add_argument('-s', '--subset', type=str, help='Subset. Choices: val, test', choices=['val', 'test'], required=True)
    parser.add_argument('--device', type=int, default=0, help="GPU device to use, default is 0")
    args = parser.parse_args()

    dataset_path = args.path
    device = args.device
    subset = args.subset

    net = SFD_NET(device=device)

    if subset == 'val':
        wider_face = sio.loadmat('./output/WIDER_FACE/wider_face_val.mat')
    else:
        wider_face = sio.loadmat('./output/WIDER_FACE/wider_face_test.mat')
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    save_path = './output/WIDER_FACE/eval_tools/sfd_' + subset + '/'
    for index, event in enumerate(event_list):
        filelist = file_list[index][0]
        if not os.path.exists(save_path + event[0][0]):
            os.makedirs(save_path + event[0][0])

        for num, file in enumerate(filelist):
            im_name = file[0][0]
            Image_Path = dataset_path + event[0][0] + '/' + im_name[:] + '.jpg'
            image = caffe.io.load_image(Image_Path)

            max_im_shrink = (0x7fffffff / 800.0 / (image.shape[0] * image.shape[1] * image.shape[2])) ** 0.5 # the max size of input blob for caffe

            shrink = max_im_shrink if max_im_shrink < 1 else 1

            det0 = net.detect(image, shrink)
            det1 = flip_test(net, image, shrink)    # flip test
            [det2, det3] = multi_scale_test(net, image, max_im_shrink)  #multi-scale test

            # Merge all test results via bounding box voting
            det = np.row_stack((det0, det1, det2, det3))
            dets = bbox_vote(det)

            f = open(save_path + event[0][0].encode('utf-8') + '/' + im_name + '.txt', 'w')
            write_to_txt(f, dets)
            print('event:%d num:%d' % (index + 1, num + 1))
