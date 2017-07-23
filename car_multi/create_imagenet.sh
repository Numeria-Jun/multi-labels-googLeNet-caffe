#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

TRAIN_DATA_ROOT=/data1/liangdas/multi_labels/caffe_script/car_multi/data
TOOLS=/root/caffe/build/tools

VAL_DATA_ROOT=/data1/liangdas/multi_labels/caffe_script/car_multi/data
# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."
#--resize_height=227  --resize_width=227 ZnCar/ ZnCar/Label.txt ZnCarTrainImage ZnCarTrainLabel 2
#rm -r $TRAIN_DATA_ROOT/imagenet_train_lmdb


GLOG_logtostderr=1 $TOOLS/convert_multilabel \
    --resize_height=227 \
    --resize_width=227 \
    $TRAIN_DATA_ROOT/ \
    $TRAIN_DATA_ROOT/new_label_train.txt \
    $TRAIN_DATA_ROOT/imagenet_train_lmdb \
    $TRAIN_DATA_ROOT/imagenet_train_label \
    3
echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_multilabel \
    --resize_height=227 \
    --resize_width=227 \
    $TRAIN_DATA_ROOT/ \
    $TRAIN_DATA_ROOT/new_label_test.txt \
    $TRAIN_DATA_ROOT/imagenet_test_lmdb \
    $TRAIN_DATA_ROOT/imagenet_test_label \
    3

echo "Done."
