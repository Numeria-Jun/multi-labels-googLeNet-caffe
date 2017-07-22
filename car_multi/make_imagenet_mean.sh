#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#DATA=/hme/parallels/caffe/car_multi
#TOOLS=/home/parallels/caffe/build/tools
DATA=/data1/liangdas/multi_labels/caffe_script/car_multi
TOOLS=/root/caffe/build/tools
$TOOLS/compute_image_mean $DATA/data/example_lmdb \
  $DATA/data/example_mean.binaryproto
#$TOOLS/compute_image_mean $DATA/data/imagenet_train_lmdb \
#  $DATA/data/imagenet_train_mean.binaryproto
#$TOOLS/compute_image_mean $DATA/data/imagenet_test_lmdb \
#  $DATA/data/imagenet_test_mean.binaryproto
echo "Done."
