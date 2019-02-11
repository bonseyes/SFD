
stage=train
python gen_model.py --stage $stage --lmdb /media/sdf/COCO/2014/lmdb/coco_train_lmdb --label-map data/coco/labelmap_coco.prototxt --class-num 81 > $stage.prototxt

stage=test
python gen_model.py --stage $stage --lmdb /media/sdf/COCO/2014/lmdb/coco_minival_lmdb --label-map data/coco/labelmap_coco.prototxt --class-num 81 > $stage.prototxt

stage=deploy
python gen_model.py --stage $stage --lmdb /media/sdf/COCO/2014/lmdb/coco_minival_lmdb --label-map data/coco/labelmap_coco.prototxt --class-num 81 > $stage.prototxt

