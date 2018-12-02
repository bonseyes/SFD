#!/bin/sh
# Generate Caffe based files for training and testing models

generate_model()
{
  STAGE=$1
  LMDB=$2
  LABELMAP=$3
  CLASSNUM=$4
  OUTFILE=$STAGE + ".prototxt"

  echo "Genreating " $STAGE
  python gen_model_fp16.py --stage $STAGE --lmdb $LMDB --label-map $LABELMAP --class-num $CLASSNUM > $OUTFILE
}

LMDB=/home/ubuntu/data/WIDER_FACE/lmdb/WIDER_FACE_train_lmdb_NVCaffe
LABELMAP=/home/ubuntu/data/WIDER_FACE/labelmap_wider.prototxt
CLASSNUM=2

# Generate solver
generate_model solver $LMDB $LABELMAP $CLASSNUM

# Generate train
generate_model train $LMDB $LABELMAP $CLASSNUM

# Generate test
generate_model test $LMDB $LABELMAP $CLASSNUM

# Generate deployment with original layers
generate_model deploy-orig $LMDB $LABELMAP $CLASSNUM

# Generate deployment with compitable with Caffe
generate_model deploy-compat $LMDB $LABELMAP $CLASSNUM
