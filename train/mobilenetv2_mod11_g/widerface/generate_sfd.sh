
stage=train
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/wider+nir-rgb+gray/lmdb/wider+nir-rgb+gray_train_lmdb_NVCaffe --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

stage=test
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/WIDER_FACE/lmdb/WIDER_FACE_val_lmdb_NVCaffe --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

stage=deploy
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/WIDER_FACE/lmdb/WIDER_FACE_val_lmdb_NVCaffe --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

