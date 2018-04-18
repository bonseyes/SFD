import argparse
import matplotlib.pyplot as plt
import os.path
from numpy import arange
from utils.io_utils import parse_detected_faces, parse_gt_faces, parse_baseline
from utils.pr_utils import compute_prec_rec, correct_pr_curve, compute_ap, normalize_scores


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

    python2.7 plot_wider.py -p -p ../output/WIDER_FACE/eval_tools/pred_orig\
           -m ../output/WIDER_FACE/eval_tools/ground_truth\
           -b ../output/WIDER_FACE/eval_tools/plot/baselines/Val/setting_int
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
        ap = compute_ap(precision, recall)
        precision, recall, _ = correct_pr_curve(precision, recall)
        overall_results[level]['trained_model-{0:.2f}'.format(ap)] = (recall, precision)
        for baseline in TO_PLOT:
            path = os.path.join(args.baselines, baseline, 'wider_pr_info_{}_{}_val.mat'.format(baseline, level))
            precision, recall = parse_baseline(path)
            overall_results[level][baseline] = (recall, precision)

    # Scale
    scale_levels = ['small', 'medium', 'large']
    scale_results = {level: {} for level in scale_levels}
    for level in scale_levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_easy_val.mat'), assesment='scale', level=level)
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        ap = compute_ap(precision, recall)
        precision, recall, _ = correct_pr_curve(precision, recall)
        scale_results[level]['trained_model-{0:.2f}'.format(ap)] = (recall, precision)

    # Occlusion 
    occlusion_levels = ['none', 'partial', 'heavy']
    occlusion_results = {level: {} for level in occlusion_levels}
    for level in occlusion_levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_easy_val.mat'), assesment='occlusion', level=level)
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        ap = compute_ap(precision, recall)
        precision, recall, _ = correct_pr_curve(precision, recall)
        occlusion_results[level]['trained_model-{0:.2f}'.format(ap)] = (recall, precision)

    # Pose
    pose_levels = ['typical', 'extreme']
    pose_results = {level: {} for level in pose_levels}
    for level in pose_levels:
        gt_faces, gt_keep = parse_gt_faces(os.path.join(args.matlab_faces, 'wider_easy_val.mat'), assesment='pose', level=level)
        precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
        ap = compute_ap(precision, recall)
        precision, recall, _ = correct_pr_curve(precision, recall)
        pose_results[level]['trained_model-{0:.2f}'.format(ap)] = (recall, precision)

    # Finally, plot everything
    length = 4
    ticks = arange(0, 1.2, 0.2) 
    f, axes = plt.subplots(len(overall_levels), length)

    for i, level in enumerate(overall_levels):
        for model in overall_results[level]:
            axes[i, 0].plot(*(overall_results[level][model]), label=model)
            axes[i, 0].set_xticks(ticks)
            axes[i, 0].set_yticks(ticks)
            axes[i, 0].set_xlabel("Recall - {}".format(level))
            axes[i, 0].set_ylabel("Precision")
            axes[i, 0].grid(which='both')
            axes[i, 0].legend()
    axes[0, 0].set_title("Overall")

    for i, level in enumerate(scale_levels):
        for model in scale_results[level]:
            axes[i, 1].plot(*(scale_results[level][model]), label=model)
            axes[i, 1].set_xticks(ticks)
            axes[i, 1].set_yticks(ticks)
            axes[i, 1].set_xlabel("Recall - {}".format(level))
            axes[i, 1].set_ylabel("Precision")
            axes[i, 1].grid(which='both')
            axes[i, 1].legend()
    axes[0, 1].set_title("Scale")

    for i, level in enumerate(occlusion_levels):
        for model in occlusion_results[level]:
            axes[i, 2].plot(*(occlusion_results[level][model]), label=model)
            axes[i, 2].set_xticks(ticks)
            axes[i, 2].set_yticks(ticks)
            axes[i, 2].set_xlabel("Recall - {}".format(level))
            axes[i, 2].set_ylabel("Precision")
            axes[i, 2].grid(which='both')
            axes[i, 2].legend()
    axes[0, 2].set_title("Occlusion")

    for i, level in enumerate(pose_levels):
        for model in pose_results[level]:
            axes[i, 3].plot(*(pose_results[level][model]), label=model)
            axes[i, 3].set_xticks(ticks)
            axes[i, 3].set_yticks(ticks)
            axes[i, 3].set_xlabel("Recall - {}".format(level))
            axes[i, 3].set_ylabel("Precision")
            axes[i, 3].grid(which='both')
            axes[i, 3].legend()
    axes[0, 3].set_title("Pose")
    axes[-1, -1].axis('off')

    #plt.savefig("WIDER_metrics.pdf")
    plt.show()
