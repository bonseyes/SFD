
stage=train
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/person+face/lmdb/person+face_train_lmdb_NVCaffe --label-map /home/ubuntu/caffe-NVIDIA/data/person+face/labelmap_person+face.prototxt --class-num 3 > $stage.prototxt

stage=test
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/person+face/lmdb/person+face_val_lmdb_NVCaffe --label-map /home/ubuntu/caffe-NVIDIA/data/person+face/labelmap_person+face.prototxt --class-num 3 > $stage.prototxt

stage=deploy
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/person+face/lmdb/person+face_val_lmdb_NVCaffe --label-map /home/ubuntu/caffe-NVIDIA/data/person+face/labelmap_person+face.prototxt --class-num 3 > $stage.prototxt

