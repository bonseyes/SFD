import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from nuscenes_nviso import create_nuscenes_infos, draw_box_custom
import argparse

NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

parser = argparse.ArgumentParser(prog='NUSCENES CAMERA DATASET')
parser.add_argument('dataroot', type=str, default="NUSCENES_MINI_DATASET_ROOT", help='root path of the nuscense dataset')
parser.add_argument('version', type=str, default="v1.0-mini", help='nuscenes dataset version ["v1.0-trainval", "v1.0-test", "v1.0-mini"]')
parser.add_argument('token', type=str, default="CAM_FRONT", help='indicated which camera data among 6 cameras')
args = parser.parse_args()

root_path = args.dataroot
version = args.version
cam_token = args.token

nusc, train_scenes, val_scenes = create_nuscenes_infos(root_path=root_path, version=version)

for sample in nusc.sample:
	if sample["scene_token"] in val_scenes:

		cam_data_token = sample["data"][cam_token]
		cam_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token=cam_data_token, box_vis_level=BoxVisibility.ALL)

		names = [b.name for b in boxes]
		for i in range(len(names)):
			if names[i] in NameMapping:
				names[i] = NameMapping[names[i]]
		names = np.array(names)

		im = cv2.imread(cam_path)
		for label, box in zip(names, boxes):
			# label is the class label according to above mapping
			# box is the 3d position information
			corners = view_points(points=box.corners(),view=camera_intrinsic, normalize=True)[:2,:]
			draw_box_custom(im=im, corners=corners)

		cv2.imshow("img", im)
		cv2.waitKey(1)
