from collections import defaultdict
import argparse
import numpy as np
import os
import os.path
import scipy.io as sio


def parse_det_file(path):
    """
    Given a path for a single detection file, 
    returns the list of faces in it:

    The returned list has shape (detected_faces, 5),
    where the columns correspond to (xmin, ymin, width, height, score) 
    The faces are sorted in descending order (i.e. bigger score first)
    """
    with open(path, 'r') as f:
        lines = [[float(i) for i in row.strip().replace('\n', '').split(' ')] for row in f.readlines()[2:]]

    faces = np.array(lines, dtype=float)
    sorted_faces = faces#[faces[:, 4].argsort()[::-1]]
    sorted_faces[:, 2] = sorted_faces[:, 0] + sorted_faces[:, 2]
    sorted_faces[:, 3] = sorted_faces[:, 1] + sorted_faces[:, 3]
    return sorted_faces


def parse_detected_faces(path):
    """
    Given a root folder where the detected faces (obtained, for example, with test_wider.py),
    parse the files and return a dict with all the data

    The root folder must have the following structure:

    root_folder/
        0--Parade/
            0_Parade_marchingband_1_1004.txt
            0_Parade_marchingband_1_1045.txt
            .
            .
            .
        1--Handshaking/
            1_Handshaking_Handshaking_1_107.txt
            1_Handshaking_Handshaking_1_134.txt
            .
            .
            .
        .
        .
        .

    Returns:
    {
        '0--Parade': {
            '0_Parade_marchingband_1_1004': [
                                             [8.53600000e+02, ..., 6.10000000e-02],
                                             [1.84000000e+02, ..., 6.00000000e-02],
                                             ...
                                            ],
            '0_Parade_marchingband_1_1045': [...]
        },
        ...
    }

    """
    result = {}
    for root, dirs, files in os.walk(path):
        category = os.path.basename(root)
        faces = files
        result[category] = {f.replace('.txt', ''): parse_det_file(os.path.join(path, category, f)) for f in faces}

    return result


def normalize_scores(det_faces):
    """
    Normalize scores between 0 and 1
    """
    max_score = 0
    min_score = 100
    for event in det_faces:
        for img in det_faces[event]:
            for det in det_faces[event][img]:
                max_score = max(det[4], max_score)
                min_score = min(det[4], min_score)

    for event in det_faces:
        for img in det_faces[event]:
            det_faces[event][img][:, 4] = (det_faces[event][img][:, 4] - min_score) / float(max_score - min_score)

    return det_faces
            

def parse_gt_faces(path):
    """
    Given a Matlab .mat file with the corresponding groundtruth faces, 
    this function loads it and returns a dictionary like the one in
    parse_detected_faces().

    The .mat files used here can be found in the original 
    Evaluation tools of WIDER dataset: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
    (eval_tools.zip, under the folder ./ground_truth)

    I recommend loading the .mat file in a Python interpreter to actually 
    understand its structure and therefore this code.
    """
    mat_faces = sio.loadmat(path)
    categories = [str(event[0]) for event in mat_faces['event_list'].flatten()]
    files = [[str(f[0]) for f in files.flatten()] for files in mat_faces['file_list'].flatten()]
    # Retrieve the list of gt boxes to keep. Matlab uses 1-based indexes, so transform them to Python indexes
    keep_index = [[gt_list[0].flatten() - 1 for gt_list in event[0]] for event in mat_faces['gt_list']]
    faces = [[[face for face in image[0]] for image in event[0]] for event in mat_faces['face_bbx_list']]

    files_faces = [{k: np.array(v) for k, v in zip(fil, faces[i])} for i, fil in enumerate(files)]
    files_keep = [{k: np.array(v) for k, v in zip(fil, keep_index[i])} for i, fil in enumerate(files)]
    # Transform width, height into ymax, xmax respectively
    for event in files_faces:
        for img in event:
            faces = event[img]
            if faces.size != 0:
                faces[:, 2] = faces[:, 0] + faces[:, 2]
                faces[:, 3] = faces[:, 1] + faces[:, 3]
                faces = faces.astype(float)

    result = dict(zip(categories, files_faces))
    result_keep = dict(zip(categories, files_keep))
    return result, result_keep


def IoU(rect1, rect2):
    """
    rect1 and rect2 are np.arrays of shape (4,)
    representing (xmin, ymin, xmax, ymax)
    """
    left1, top1, right1, bottom1 = rect1.astype(float)
    left2, top2, right2, bottom2 = rect2.astype(int)
    xmin = int(round(max(left1, left2)))
    xmax = int(round(min(right1, right2)))
    ymin = int(round(max(top1, top2)))
    ymax = int(round(min(bottom1, bottom2)))
    w = max(xmax - xmin + 1, 0) 
    h = max(ymax - ymin + 1, 0) 
    inter_area = w * h
    rect1_area = (right1 - left1 + 1) * (bottom1 - top1 + 1)
    rect2_area = (right2 - left2 + 1) * (bottom2 - top2 + 1)
    union_area = int(round(rect1_area + rect2_area - inter_area))
    return inter_area / float(union_area)


def compute_overlaps(rect, gt):
    """
    Given a single rectangle and a list of groundtruth rectangles,
    compute a list of the IoU for each gt rectangle
    """
    # TODO: vectorize
    ious = []
    for gt_rect in gt:
        ious.append(IoU(rect[:-1], gt_rect))
    return np.array(ious)


def compute_pr(dets, gt_dets, gt_keep):
    """
    dets, gt_dets are np.arrays(dtype=float) that have
    shape (num_detections, 4)

    This function assumes `dets` has the detections sorted
    in descending order according to a score.

    Returns count of cumulative TP (pred_recalls), and a list of proposals that need to be ignored
    """
    # recalls keeps track of all the gt boxes that became
    # TP detections. We do not count the ignored gt boxes
    # in the metric. If a box was detected, it has 1, otherwise 
    # its value in this array will be 0.
    recalls = np.zeros(len(gt_dets)) 
    # Counts cumulative TP for each detection
    pred_recalls = np.zeros(len(dets))
    # proposals keeps track of all the detections:
    # - The ones that overlap with gt boxes that are ignored at this 
    #   difficulty level (-1)
    # - The ones that are either TP or FP (1)
    proposals = np.ones(len(dets))

    for i, det in enumerate(dets):
        ious = compute_overlaps(det, gt_dets)
        idx = np.argmax(ious)
        max_iou = ious[idx]
        if max_iou >= 0.5:
            if idx not in gt_keep:
                recalls[idx] = -1
                proposals[i] = -1
            elif recalls[idx] == 0:
                recalls[idx] = 1
        true_positives = np.sum(recalls == 1)
        pred_recalls[i] = true_positives

    return pred_recalls, proposals


def compute_tp_fp(dets, pred_recalls, proposals):
    thresholds = 1000
    interp_pr = np.zeros((thresholds, 2))

    for i, r in enumerate(np.arange(0, 1, 1 / float(thresholds))):
    #for i, r in enumerate(np.arange(0.999, 0.0, -1 / float(thresholds))):
        t = i + 1
    #for t in range(1, thresholds+1):
        #r = 1 - t / float(thresholds)
        idx = np.where(dets[:, 4] >= r)[0]
        if idx.size == 0:
            interp_pr[t - 1, 0] = 0
            interp_pr[t - 1, 1] = 0
        else:
            min_idx = idx[-1]
            # amount of TP and FP for this threshold, for this image
            interp_pr[t - 1, 0] = np.sum(proposals[:min_idx + 1] == 1)
            # amount of TP for this threshold, for this image
            interp_pr[t - 1, 1] = pred_recalls[min_idx]

    return interp_pr


#def compute_tp_fp(dets, pred_recalls, proposals):
#    thresholds = 1000
#    interp_pr = np.zeros((thresholds, 2))
#
#    for t in range(1, thresholds+1):
#        r = 1 - t / float(thresholds)
#        idx = np.where(dets[:, 4] >= r)[0]
#        if idx.size == 0:
#            interp_pr[t - 1, 0] = 0
#            interp_pr[t - 1, 1] = 0
#        else:
#            min_idx = idx[-1]
#            # amount of TP and FP for this threshold, for this image
#            interp_pr[t - 1, 0] = len(np.where(proposals[:min_idx + 1] == 1)[0])
#            # amount of TP for this threshold, for this image
#            interp_pr[t - 1, 1] = pred_recalls[min_idx]
#
#    return interp_pr


def compute_prec_rec(det_faces, gt_faces, gt_keep):
    """
    Given dicts of detected faces and groundtruth faces
    compute the mAP.

    The dicts are generated by parse_detected_faces and parse_gt_faces 
    respectively.
    """
    all_interp_pr = []
    count = 0
    n_imgs = 0
    for event in det_faces:
        for img in det_faces[event]:
            n_imgs += 1
            img_faces = det_faces[event][img]#[:, :-1]
            img_gt_faces = gt_faces[event][img]
            img_gt_keep = gt_keep[event][img]
            count += len(img_gt_keep)

            if img_faces.size == 0 or img_gt_faces.size == 0:
                continue

            #print(img)
            #if "0_Parade_marchingband_1_465" in img:
            #    import ipdb; ipdb.set_trace()
            pred_recalls, proposals = compute_pr(img_faces, img_gt_faces, img_gt_keep)
            interp_pr = compute_tp_fp(img_faces, pred_recalls, proposals)
            all_interp_pr.append(interp_pr)

    thresholds = 1000
    tpfp = np.zeros(thresholds)
    tp = np.zeros(thresholds)
    for i in range(n_imgs):
        tpfp = tpfp + all_interp_pr[i][:, 0]
        tp = tp + all_interp_pr[i][:, 1]

    prec = tp / tpfp
    recall = tp / float(count)
    return prec, recall


def correct_pr_curve(precision, recall):
    """
    We will consider a corrected PR curve where for 
    each point (p, r), if there exists another point
    (p', r') such that p' > p and r' >= r, we set p
    to the maximum of those points
    """
    off_pre = np.concatenate(([0], precision[::-1] , [0]))
    off_rec = np.concatenate(([0], recall[::-1], [1]))

    for i in range(len(off_pre) - 2, -1, -1):
        off_pre[i] = max(off_pre[i], off_pre[i + 1])

    idx = np.where(off_rec[1:] != off_rec[:-1])[0] + 1

    return off_pre, off_rec, idx


def compute_ap(precision, recall):
    precision, recall, idx = correct_pr_curve(precision, recall)
    return np.sum((recall[idx] - recall[idx - 1]) * precision[idx])


if __name__ == '__main__':
    """
    Usage:

    python2.7 test_wider_mAP.py -p WIDER_FACE/eval_tools_old-version/sfd_val\
           -m WIDER_FACE/eval_tools_old-version/ground_truth/wider_easy_val.mat
    """
    parser = argparse.ArgumentParser(description='Code to check mAP metric for WIDER FACE obtained with SFD.')
    parser.add_argument('-p', '--path', type=str, help='Folder where the detected faces (validation set) are.', required=True)
    parser.add_argument('-m', '--matlab-faces', type=str, help='.mat file with the groundtruth faces. Can be easy, medium or hard.', required=True)
    args = parser.parse_args()

    dets_path = args.path
    gt_path = args.matlab_faces

    det_faces = parse_detected_faces(dets_path)
    det_faces = normalize_scores(det_faces)
    gt_faces, gt_keep = parse_gt_faces(gt_path)
    precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
    ap = compute_ap(precision, recall)
    print("WIDERFACE AP: {0:.2f}".format(ap * 100))
