import argparse
import matplotlib as plt
import os.path
from utils.io_utils import parse_detected_faces, parse_gt_faces, parse_baseline
from utils.pr_utils import compute_prec_rec, compute_ap, normalize_scores


"""
NOTE:
    Most of the baselines used here are already into the WIDER FACE eval_tools.
    If you dont have eval_tools downloaded, check Test_instructions.md in this repository.

    However, PyramidBox and SFD-C seem to be missing. You can download them 
    and place them inside ../output/WIDER_FACE/eval_tools/plot/baselines/Val/setting_int/
    under the names PyramidBox and SFD-C. This should be automated in a future.

    PyramidBox baseline: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/baselines/Val/setting_int/PyramidBox.zip
    SFD-C baseline: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/baselines/Val/setting_int/SFD-C.zip
"""


# Omit ACF as it only has 100 points in the pr curve
TO_PLOT = ['PyramidBox', 'SFD-C', 'multitask-cascade-cnn', 'Faceness', 'SSH']


if __name__ == '__main__':
    """
    Usage:

    python2.7 plot_wider.py -p WIDER_FACE/eval_tools/sfd_val\
           -m ../output/WIDER_FACE/eval_tools/plot/baselines/Val/setting_int/
    """
    parser = argparse.ArgumentParser(description='Code to check mAP metric for WIDER FACE obtained with SFD.')
    parser.add_argument('-p', '--path', type=str, help='Folder where the detected faces (validation set) are.', required=True)
    parser.add_argument('-m', '--matlab-faces', type=str, help='Path where the easy, medium and hard .mat files with the groundtruth faces are.', required=True)
    parser.add_argument('-b', '--baselines', type=str, help='Path where the .mat files with baselines are.', required=True)
    args = parser.parse_args()

    dets_path = args.path
    baselines = [os.path.join(args.baselines, baseline) for baseline in TO_PLOT]

    det_faces = parse_detected_faces(dets_path)
    det_faces = normalize_scores(det_faces)

    levels = ['easy', 'medium', 'hard']
    results = {level: {} for level in levels}

    for level in levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_{}_val.mat'.format(level)))
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        results[level]['trained_model'] = (precision, recall)
        for baseline in TO_PLOT:
            path = os.path.join(args.baselines, baseline, 'wider_pr_info_{}_{}_val.mat'.format(baseline, level))
            precision, recall = parse_baseline(path)
            results[level][baseline] = (precision, recall)
