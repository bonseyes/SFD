import argparse
import matplotlib.pyplot as plt
import os.path
from utils.io_utils import parse_detected_faces, parse_gt_faces, parse_baseline
from utils.pr_utils import compute_prec_rec, compute_ap, normalize_scores


"""
NOTE:
    Most of the baselines used here are already into the WIDER FACE eval_tools.
    If you dont have eval_tools downloaded, check Test_instructions.md in this repository.

    However, PyramidBox and SFD-C seem to be missing in eval_tools. You can download their results
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


    # Compute PR curves for Overall, Scale, Occlusion and Pose
    overall_levels = ['easy', 'medium', 'hard']
    overall_results = {level: {} for level in overall_levels}

    # Overall
    for level in overall_levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_{}_val.mat'.format(level)))
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        overall_results[level]['trained_model'] = (precision, recall)
        for baseline in TO_PLOT:
            path = os.path.join(args.baselines, baseline, 'wider_pr_info_{}_{}_val.mat'.format(baseline, level))
            precision, recall = parse_baseline(path)
            overall_results[level][baseline] = (recall, precision)

    # Scale
    scale_levels = ['small', 'medium', 'large']
    scale_results = {level: {} for level in scale_levels}
    for level in scale_levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_easy_val.mat'.format(level)), assesment='scale', level=level)
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        scale_results[level]['trained_model'] = (recall, precision)

    # Occlusion 
    occlusion_levels = ['none', 'partial', 'heavy']
    occlusion_results = {level: {} for level in occlusion_levels}
    for level in occlusion_levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_easy_val.mat'.format(level)), assesment='occlusion', level=level)
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        occlusion_results[level]['trained_model'] = (recall, precision)

    # Pose
    pose_levels = ['typical', 'extreme']
    pose_results = {level: {} for level in pose_levels}
    for level in pose_levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_easy_val.mat'.format(level)), assesment='pose', level=level)
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        pose_results[level]['trained_model'] = (recall, precision)

    # Finally, plot everything
    plt.figure(1)
    prefix = '34'
    length = 4

    start = 1
    for i, level in enumerate(overall_levels):
        plt.subplot(int(prefix + str(start + i * length)))
        for model in overall_results[level]:
            plt.plot(*(overall_results[level][model]))
    plt.grid(True)

    start = 2
    for i, level in enumerate(scale_levels):
        plt.subplot(int(prefix + str(start + i * length)))
        for model in scale_results[level]:
            plt.plot(*(scale_results[level][model]))
    plt.grid(True)

    start = 3
    for i, level in enumerate(occlusion_levels):
        plt.subplot(int(prefix + str(start + i * length)))
        for model in occlusion_results[level]:
            plt.plot(*(occlusion_results[level][model]))
    plt.grid(True)

    start = 4
    for i, level in enumerate(pose_levels):
        plt.subplot(int(prefix + str(start + i * length)))
        for model in pose_results[level]:
            plt.plot(*(pose_results[level][model]))
    plt.grid(True)
    plt.show()
