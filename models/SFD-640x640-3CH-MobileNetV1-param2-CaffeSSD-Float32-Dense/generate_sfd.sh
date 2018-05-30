
stage=train

python gen_SFD-MobileNet_3CH.py --stage $stage --lmdb examples/WIDER_FACE/WIDER_FACE_train_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt


stage=test

python gen_SFD-MobileNet_3CH.py --stage $stage --lmdb examples/WIDER_FACE/WIDER_FACE_val_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

stage=deploy

python gen_SFD-MobileNet_3CH.py --stage $stage --lmdb examples/WIDER_FACE/WIDER_FACE_val_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

