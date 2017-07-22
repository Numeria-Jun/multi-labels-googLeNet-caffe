#!/usr/bin/env sh

CAFFE_ROOT="/home/will/deepLearning/caffe-ssd"
CAFFE_TOOLS="/home/will/deepLearning/caffe-ssd/build/tools"
LMDB_FOLDER="/sata2/lmdb_folder"
TRAIN_ROOT_INPUT=$1
RESIZE_WIDTH=$2
RESIZE_HEIGHT=$3


#1. create train data
rm -rf ${LMDB_FOLDER}"/img_train_lmdb"
echo "create validate data"
GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_ROOT_OUTPUT"/" \
    ${TRAIN_ROOT_OUTPUT}"/label_train.txt" \
    ${LMDB_FOLDER}"/img_train_lmdb"
    
#5. creat mean file
echo "creat mean file"
rm -rf ${TRAIN_ROOT_OUTPUT}"/img_train_mean.binaryproto"
$CAFFE_TOOLS/compute_image_mean \
	${LMDB_FOLDER}"/img_train_lmdb" \
	${LMDB_FOLDER}"/img_train_mean.binaryproto"

