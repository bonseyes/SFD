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
    sorted_faces = faces[faces[:, 4].argsort()[::-1]]
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
    faces = [[[list(face) for face in image[0]] for image in event[0]] for event in mat_faces['face_bbx_list']]
    files_faces = [{k: np.array(v) for k, v in zip(fil, faces[i])} for i, fil in enumerate(files)]
    # Transform width, height into ymax, xmax respectively
    for event in files_faces:
        for img in event:
            faces = event[img]
            faces[:, 2] = faces[:, 0] + faces[:, 2]
            faces[:, 3] = faces[:, 1] + faces[:, 3]
            faces = faces.astype(float)

    result = dict(zip(categories, files_faces))
    return result


def IoU(rect1, rect2):
    """
    rect1 and rect2 are np.arrays of shape (4,)
    representing (xmin, ymin, xmax, ymax)
    """
    left1, top1, right1, bottom1 = rect1.astype(float)
    left2, top2, right2, bottom2 = rect2.astype(float)
    xmin = max(left1, left2)
    xmax = min(right1, right2)
    ymin = max(top1, top2)
    ymax = min(bottom1, bottom2)
    w = max(xmax - xmin + 1, 0) 
    h = max(ymax - ymin + 1, 0) 
    inter_area = w * h
    rect1_area = (right1 - left1 + 1) * (bottom1 - top1 + 1)
    rect2_area = (right2 - left2 + 1) * (bottom2 - top2 + 1)
    union_area = rect1_area + rect2_area - inter_area
    return inter_area / float(union_area)


def compute_prec_recall(dets, gt_dets):
    """
    dets, gt_dets are np.arrays(dtype=float) that have
    shape (num_detections, 4)

    This function assumes `dets` has the detections sorted
    in descending order according to a score. The computation of
    precision/recall is then done for each rank.
    """
    already_detected = []
    true_positives = 0
    false_positives = 0
    precisions = []
    recalls = []
    for det in dets:
        for i, gt in enumerate(gt_dets):
            iou = IoU(det, gt)
            if iou > 0.5:
                if i not in already_detected:
                    true_positives += 1
                    already_detected.append(i)
                else:
                    # The bbox was already detected before with >= confidence
                    # This is then a false positive
                    false_positives += 1
                break
        else:
            # All the groundtruth bboxes were checked and the IoU was below 0.5 in all cases
            # i.e. this is a False Positive
            false_positives += 1
        # If there are groundtruth boxes that were not
        # detected, they are false negatives
        false_negatives = len(gt_dets) - true_positives

        # Compute precision and recall for this rank
        precision = true_positives / float(true_positives + false_positives)
        recall = true_positives / float(true_positives + false_negatives)
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)


def compute_interpolated_AP(precisions, recalls):
    """
    Given a list of precisions and recalls for a single
    image (computed cumulatively for the detections
    sorted in descending order), compute the average
    precision at a set of eleven equally spaced recall 
    levels [0, 0.1 , ..., 1]

    For more details check the Section 4.2 of "The PASCAL Visual Object 
    Classes (VOC) Challenge" report: 
    http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
    """
    interp_precs = [] 
    for r in np.arange(0, 1.1, 0.1):
        idx = np.where(recalls > r)[0]
        if len(idx) == 0:
            interp_precs.append(0)
            continue
        min_idx = idx[recalls[idx].argmin()]
        interp_precs.append(precisions[min_idx])

    interp_precs = np.array(interp_precs)
    assert len(interp_precs) == 11
    return interp_precs.mean()


def compute_mAP(det_faces, gt_faces):
    """
    Given dicts of detected faces and groundtruth faces
    compute the mAP.

    The dicts are generated by parse_detected_faces and parse_gt_faces 
    respectively.
    """
    all_AP = []
    for event in det_faces:
        for img in det_faces[event]:
            img_faces = det_faces[event][img][:, :-1]
            img_gt_faces = gt_faces[event][img]
            precisions, recalls = compute_prec_recall(img_faces, img_gt_faces)
            interp_AP = compute_interpolated_AP(precisions, recalls)
            all_AP.append(interp_AP)

    all_AP = np.array(all_AP)
    return all_AP.mean()


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
    gt_faces = parse_gt_faces(gt_path)
    mAP = compute_mAP(det_faces, gt_faces)
    print("WIDERFACE mAP: {0:.2f}".format(mAP * 100))
