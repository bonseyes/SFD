
import json
import time
from pathlib import Path
import numpy as np
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
#from nuscenes.utils.geometry_utils import view_points, BoxVisibility

def draw_box_custom(im, corners, colors = ((0, 0, 255), (0, 0, 255), (0, 0, 255))):
	
	linewidth = 2
	def draw_rect(selected_corners, color):
		prev = selected_corners[-1]
		for corner in selected_corners:
			cv2.line(im,
					(int(prev[0]), int(prev[1])),
					(int(corner[0]), int(corner[1])),
					color, linewidth)
			prev = corner
	# Draw the sides
	for i in range(4):
		cv2.line(im,
			(int(corners.T[i][0]), int(corners.T[i][1])),
			(int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
			colors[2][::-1], linewidth)

	# Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
	draw_rect(corners.T[:4], colors[0][::-1])
	draw_rect(corners.T[4:], colors[1][::-1])

	# Draw line indicating the front
	center_bottom_forward = np.mean(corners.T[2:4], axis=0)
	center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
	cv2.line(im,
			(int(center_bottom[0]), int(center_bottom[1])),
			(int(center_bottom_forward[0]), int(center_bottom_forward[1])),
			colors[0][::-1], linewidth)

# filter exist scenes. you may only download part of dataset.
def _get_available_scenes(nusc):
	available_scenes = []

	print("total scene num:", len(nusc.scene))
	for scene in nusc.scene:
		scene_token = scene["token"]
		scene_rec = nusc.get('scene', scene_token)
		sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
		sd_rec = nusc.get('sample_data', sample_rec['data']["CAM_FRONT"])
		has_more_frames = True
		scene_not_exist = False
		while has_more_frames:
			image_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
			if not Path(image_path).exists():
				scene_not_exist = True
				break
			else:
				break
			if not sd_rec['next'] == "":
				sd_rec = nusc.get('sample_data', sd_rec['next'])
			else:
				has_more_frames = False
		if scene_not_exist:
			continue
		available_scenes.append(scene)
	print("exist scene num:", len(available_scenes))
	return available_scenes

def create_nuscenes_infos(root_path, version):

	nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

	available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
	assert version in available_vers
	
	if version == "v1.0-trainval":
		train_scenes = splits.train
		val_scenes = splits.val
	elif version == "v1.0-test":
		train_scenes = splits.test
		val_scenes = []
	elif version == "v1.0-mini":
		train_scenes = splits.mini_train
		val_scenes = splits.mini_val
	else:
		raise ValueError("unknown")
	
	test = "test" in version
	
	root_path = Path(root_path)
	print(root_path)

	# filter exist scenes.
	available_scenes = _get_available_scenes(nusc)
	available_scene_names = [s["name"] for s in available_scenes]

	train_scenes = list(
		filter(lambda x: x in available_scene_names, train_scenes))

	val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

	train_scenes = set([
		available_scenes[available_scene_names.index(s)]["token"]
		for s in train_scenes
	])
	val_scenes = set([
		available_scenes[available_scene_names.index(s)]["token"]
		for s in val_scenes
	])
	if test:
		print(f"test scene: {len(train_scenes)}")
	else:
		print(
			f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

	return nusc, train_scenes, val_scenes