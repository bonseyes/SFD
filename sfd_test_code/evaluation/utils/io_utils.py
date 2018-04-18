import h5py
import numpy as np
import os
import os.path
import scipy.io as sio


"""
Utilities to load detected faces and groundtruth faces into memory
"""


def load_mat_file(path):
    """
    Given a .mat file, loads it into memory
    """
    return sio.loadmat(path)


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
    sorted_faces = faces
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


def parse_gt_faces(mat_faces, assesment='', level=''):
    """
    Given a preloaded Matlab .mat file with the corresponding groundtruth faces, 
    this function returns a dictionary like the one in
    parse_detected_faces().

    The .mat files used here can be found in the original 
    Evaluation tools of WIDER dataset: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
    (eval_tools.zip, under the folder ./ground_truth)

    I recommend loading the .mat file in a Python interpreter to actually 
    understand its structure and therefore this code.
    """
    categories = [str(event[0]) for event in mat_faces['event_list'].flatten()]
    files = [[str(f[0]) for f in files.flatten()] for files in mat_faces['file_list'].flatten()]
    faces = [[[face for face in image[0]] for image in event[0]] for event in mat_faces['face_bbx_list']]
    # Retrieve the list of gt boxes to keep. Matlab uses 1-based indexes, so transform them to Python indexes when necessary
    # TODO: vectorize this
    if assesment == 'pose':
        labels = {'typical': 0, 'extreme': 1}
        keep_index = [[np.where(pose_list[0].flatten() == labels[level])[0] for pose_list in event[0]] for event in mat_faces['pose_label_list']]
        # As per Section 5.1 of the Wider Face Technical Report, we'll test faces that are at least 30px in height and are un-occluded
        height_index = [[np.where(np.array(im)[:, 3] >= 30)[0] for im in event] for event in faces]
        unoccluded_index = [[np.where(occl_list[0].flatten() == 0)[0] for occl_list in event[0]] for event in mat_faces['occlusion_label_list']]
        inters_index = [[np.array(list(set(im).intersection(set(unoccluded_index[i][j])))) for j, im in enumerate(event)] for i, event in enumerate(height_index)]
        keep_index = [[np.array(list(set(im).intersection(set(inters_index[i][j])))) for j, im in enumerate(event)] for i, event in enumerate(keep_index)]
    elif assesment == 'occlusion':
        labels = {'none': 0, 'partial': 1, 'heavy': 2}
        keep_index = [[np.where(occl_list[0].flatten() == labels[level])[0] for occl_list in event[0]] for event in mat_faces['occlusion_label_list']]
        # As per Section 5.1 of the Wider Face Technical Report, we'll test faces that are at least 30px in height
        height_index = [[np.where(np.array(im)[:, 3] >= 30)[0] for im in event] for event in faces]
        keep_index = [[np.array(list(set(im).intersection(set(height_index[i][j])))) for j, im in enumerate(event)] for i, event in enumerate(keep_index)]
    elif assesment == 'scale':
        # Check height of face and select accordingly
        labels = {'small': (10, 50), 'medium': (50, 300), 'large': (300, 10000)}
        keep_index = [[np.where((np.array(im)[:, 3] >= labels[level][0]) & (np.array(im)[:, 3] < labels[level][1]))[0] for im in event] for event in faces]
        # Also, according to Section 5.1 of the Wider Face Technical Report, we'll test un-occluded faces only
        unoccluded_index = [[np.where(occl_list[0].flatten() == 0)[0] for occl_list in event[0]] for event in mat_faces['occlusion_label_list']]
        keep_index = [[np.array(list(set(im).intersection(set(unoccluded_index[i][j])))) for j, im in enumerate(event)] for i, event in enumerate(keep_index)]
    else:
        # Overall
        keep_index = [[gt_list[0].flatten() - 1 for gt_list in event[0]] for event in mat_faces['gt_list']]

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


def parse_baseline(mat_file):
    """
    Given a certain mat file with a 'pr_curve' field, 
    loads it and return the precision and recall lists.
    """
    try:
        mat = sio.loadmat(mat_file)
        precision =  mat['pr_cruve'][:, 0]
        recall =  mat['pr_cruve'][:, 1]
    except NotImplementedError:
        mat = h5py.File(mat_file, 'r')
        precision =  mat['pr_cruve'].value[0, :]
        recall =  mat['pr_cruve'].value[1, :]
    return precision, recall
