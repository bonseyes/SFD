
# NIR train
python2.7 generate-lists.py --images_dir /media/sdf/NIR-Database/images-rand --annot_dir /media/sdf/NIR-Database/annotations --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+nir-rgb+gray --mode train --seperator_images NIR-Database  --seperator_annot NIR-Database

# NIR train
python2.7 generate-lists.py --images_dir /media/sdf/NIR-Database/grayscale-images-rand --annot_dir /media/sdf/NIR-Database/annotations --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+nir-rgb+gray --mode train --seperator_images NIR-Database  --seperator_annot NIR-Database

# WIDER_FACE train
python2.7 generate-lists.py --images_dir /media/sdf/WIDER_FACE/WIDER_train/images --annot_dir /media/sdf/WIDER_FACE/WIDER_train/WIDER_train_annotations_ssd --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+nir-rgb+gray --mode train --seperator_images WIDER_FACE  --seperator_annot WIDER_FACE --wider_face

# WIDER_FACE train grayscale
python2.7 generate-lists.py --images_dir /media/sdf/WIDER_FACE/WIDER_train/images_grayscale --annot_dir /media/sdf/WIDER_FACE/WIDER_train/WIDER_train_annotations_ssd --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+nir-rgb+gray --mode train --seperator_images WIDER_FACE  --seperator_annot WIDER_FACE --wider_face

