
stage=train
python gen_model_fp16.py --classifier --stage $stage --lmdb /home/ubuntu/data/WIDER_FACE/lmdb/WIDER_FACE_train_lmdb_NVCaffe --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 1000 > $stage.prototxt

stage=test
python gen_model_fp16.py --classifier --stage $stage --lmdb /home/ubuntu/data/WIDER_FACE/lmdb/WIDER_FACE_val_lmdb_NVCaffe --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 1000 > $stage.prototxt

#stage=deploy
#python gen_model_fp16.py --classifier --stage $stage --lmdb /home/ubuntu/data/WIDER_FACE/lmdb/WIDER_FACE_val_lmdb_NVCaffe --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 1000 > $stage.prototxt

