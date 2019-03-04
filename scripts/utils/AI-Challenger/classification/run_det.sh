# validation
#python2.7 crop_augment_faces.py --data_dir=/media/sdf/AI-Challenger/original/ai_challenger_keypoint_validation_20170911 --image_dir=keypoint_validation_images_20170911 --json_file=keypoint_validation_annotations_20170911.json --faces_out_dir=/media/sdf/AI-Challenger/faces-NonFaces/faces/ai_challenger_keypoint_validation_20170911 --background_out_dir=/media/sdf/AI-Challenger/faces-NonFaces/background/ai_challenger_keypoint_validation_20170911 --confidence=0.8

# test A
python2.7 crop_augment_faces.py --data_dir=/media/sdf/AI-Challenger/original/ai_challenger_keypoint_test_a_20180103 --image_dir=keypoint_test_a_images_20180103 --json_file=keypoint_test_a_annotations_20180103.json --faces_out_dir=/media/sdf/AI-Challenger/faces-NonFaces/faces/ai_challenger_keypoint_test_a_20180103 --background_out_dir=/media/sdf/AI-Challenger/faces-NonFaces/background/ai_challenger_keypoint_test_a_20180103 --confidence=0.8

# test B
python2.7 crop_augment_faces.py --data_dir=/media/sdf/AI-Challenger/original/ai_challenger_keypoint_test_b_20180103 --image_dir=keypoint_test_b_images_20180103 --json_file=keypoint_test_b_annotations_20180103.json --faces_out_dir=/media/sdf/AI-Challenger/faces-NonFaces/faces/ai_challenger_keypoint_test_b_20180103 --background_out_dir=/media/sdf/AI-Challenger/faces-NonFaces/background/ai_challenger_keypoint_test_b_20180103 --confidence=0.8

