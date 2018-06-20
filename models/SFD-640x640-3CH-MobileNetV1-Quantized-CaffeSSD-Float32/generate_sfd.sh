
stage=train
python gen-MobileNetV1.py --stage $stage --quantization --lmdb examples/WIDER_FACE/WIDER_FACE_train_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

stage=test
python gen-MobileNetV1.py --stage $stage --quantization --lmdb examples/WIDER_FACE/WIDER_FACE_val_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

stage=deploy
python gen-MobileNetV1.py --stage $stage --quantization --lmdb examples/WIDER_FACE/WIDER_FACE_val_lmdb --label-map data/WIDER_FACE/labelmap_wider.prototxt --class-num 2 > $stage.prototxt

