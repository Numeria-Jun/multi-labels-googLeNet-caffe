#!/usr/bin/env sh
set -e

/root/caffe/build/tools/caffe train \
    --solver=/data1/liangdas/multi_labels/caffe_script/car_multi/data/solver_googlenet.prototxt \
    --weights=/data1/liangdas/multi_labels/caffe_script/car_multi/data/bvlc_googlenet.caffemodel      $@
