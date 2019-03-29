
stage=train
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/person-detection/lmdb/person-detection_train_lmdb_NVCaffe --label-map data/person-detection/labelmap_person.prototxt --class-num 2 > $stage.prototxt

stage=test
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/person-detection/lmdb/person-detection_val_lmdb_NVCaffe --label-map data/person-detection/labelmap_person.prototxt --class-num 2 > $stage.prototxt

stage=deploy
python gen_model_fp16.py --stage $stage --lmdb /media/sdf/person-detection/lmdb/person-detection_val_lmdb_NVCaffe --label-map data/person-detection/labelmap_person.prototxt --class-num 2 > $stage.prototxt

