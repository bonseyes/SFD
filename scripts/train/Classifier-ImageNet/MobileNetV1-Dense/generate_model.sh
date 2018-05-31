
# Note: change the lmdb path and other parameters to match ImageNet
stage=train
python gen-MobileNetV1.py --stage $stage --lmdb examples/WIDER_FACE/WIDER_FACE_train_lmdb --classifier --class-num 1001 > $stage.prototxt

stage=test
python gen-MobileNetV1.py --stage $stage --lmdb examples/WIDER_FACE/WIDER_FACE_val_lmdb --classifier --class-num 1001 > $stage.prototxt

stage=deploy
python gen-MobileNetV1.py --stage $stage --lmdb examples/WIDER_FACE/WIDER_FACE_val_lmdb --classifier --class-num 1001 > $stage.prototxt

