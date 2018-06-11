
stage=train
python gen_model.py --stage $stage --lmdb /home/ubuntu/data/WIDER_FACE/lmdb/WIDER_FACE_train_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

stage=test
python gen_model.py --stage $stage --lmdb /home/ubuntu/data/WIDER_FACE/lmdb/WIDER_FACE_val_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

stage=deploy
python gen_model.py --stage $stage --lmdb /home/ubuntu/data/WIDER_FACE/lmdb/WIDER_FACE_val_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

