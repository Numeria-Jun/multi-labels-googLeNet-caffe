#!/usr/bin/env sh
set -e

/root/caffe/build/tools/classification_multilabel \
/data1/liangdas/multi_labels/caffe_script/car_multi/data/deploy.prototxt \
/data1/liangdas/multi_labels/caffe_script/car_multi/data/googLenet_fine_tuning/googLenet_iter_300000.caffemodel \
/data1/liangdas/multi_labels/caffe_script/car_multi/data/example_mean.binaryproto \
label3.txt label2.txt label1.txt \
/data1/liangdas/multi_labels/caffe_script/car_multi/data/003913.jpg  $@
