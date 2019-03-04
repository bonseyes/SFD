# validation
python generate_lists_classification.py --images_dir /media/sdf/AI-Challenger/faces-NonFaces/validation-data > val.txt

shuf val.txt > val_shuffled.txt

# training
python generate_lists_classification.py --images_dir /media/sdf/AI-Challenger/faces-NonFaces/train-data > train.txt

shuf train.txt > train_shuffled.txt 
