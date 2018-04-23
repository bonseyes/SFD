import argparse
from utils.io_utils import parse_detected_faces, parse_gt_faces, load_mat_file
from utils.pr_utils import compute_prec_rec, compute_ap, normalize_scores


if __name__ == '__main__':
    """
    Usage:

    python2.7 wider_eval.py -p ../output/WIDER_FACE/eval_tools/sfd_val\
           -m ../output/WIDER_FACE/eval_tools/ground_truth/wider_easy_val.mat
    """
    parser = argparse.ArgumentParser(description='Code to check mAP metric for WIDER FACE obtained with SFD.')
    parser.add_argument('-p', '--path', type=str, help='Folder where the detected faces (validation set) are.', required=True)
    parser.add_argument('-m', '--matlab-faces', type=str, help='.mat file with the groundtruth faces. Can be easy, medium or hard.', required=True)
    args = parser.parse_args()

    dets_path = args.path
    gt_path = args.matlab_faces

    det_faces = parse_detected_faces(dets_path)
    det_faces = normalize_scores(det_faces)
    mat_file = load_mat_file(gt_path)
    gt_faces, gt_keep = parse_gt_faces(mat_file)
    precision, recall = compute_prec_rec(det_faces, gt_faces, gt_keep)
    ap = compute_ap(precision, recall)
    print("WIDERFACE AP: {0:.2f}".format(ap * 100))
