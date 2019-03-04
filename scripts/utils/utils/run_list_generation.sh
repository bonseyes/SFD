# WIDER_FACE train
python2.7 generate-lists.py --images_dir /media/sdf/WIDER_FACE/WIDER_train/images --annot_dir /media/sdf/WIDER_FACE/WIDER_train/WIDER_train_annotations_ssd --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+ --mode train --seperator_images WIDER_FACE  --seperator_annot WIDER_FACE --wider_face

# WIDER_FACE val
python2.7 generate-lists.py --images_dir /media/sdf/WIDER_FACE/WIDER_val/images --annot_dir /media/sdf/WIDER_FACE/WIDER_val/WIDER_val_annotations_ssd --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+ --mode val --seperator_images WIDER_FACE  --seperator_annot WIDER_FACE --wider_face

# MAFA all
python2.7 generate-lists.py --images_dir /media/sdf/MAFA/images --annot_dir /media/sdf/MAFA/annotations --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+ --mode train --seperator_images MAFA  --seperator_annot MAFA

# UFDD val (we use it as training)
python2.7 generate-lists.py --images_dir /media/sdf/UFDD/UFDD_val/images --annot_dir /media/sdf/UFDD/UFDD_val/annotations --out_directory /home/ubuntu/caffe-NVIDIA/data/wider+ --mode train --seperator_images UFDD --seperator_annot UFDD


