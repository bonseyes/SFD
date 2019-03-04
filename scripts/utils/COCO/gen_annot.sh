# validation set
python coco_person_json_to_txt.py --images_dir /media/sdf/COCO/2014/images/val2014 --json_dir /media/sdf/COCO/2014/Annotations/val2014 --out_dir /media/sdf/person+face-detection/COCO/val2014 --confidence 0.45

# validation set
python coco_person_json_to_txt.py --images_dir /media/sdf/COCO/2014/images/train2014 --json_dir /media/sdf/COCO/2014/Annotations/train2014 --out_dir /media/sdf/person+face-detection/COCO/train2014 --confidence 0.45

