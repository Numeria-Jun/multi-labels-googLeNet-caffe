# multi-labels-googLeNet-caffe
this repository is multi-labels classification which based on  googLeNet model with caffe .

more information you get with http://blog.csdn.net/numeria/article/details/75532998

step1:
 
Download convert_multilabel.cpp replace convert_imageset.cpp, and download classification_multilabel.cpp .
then recompile caffe with using command lines:

make clean
make all
make test
make pycaffe
make runtest

step2:

Try to manufacture your own lmdb with the example shell file which named create_imagenet.sh:

sh create_imagenet.sh

exaplain the parameters:
--resize_height=227:the image resize height
--resize_width=227:the image resize width
TRAIN_DATA_ROOT/:the directory of your data
TRAIN_DATA_ROOT/new_label_train.txt:the multi-labels files of training files ,which you can see details in car_multi/data/new_label_train.txt.
(focused on new_label_train.txt,for example 00032/003913.jpg 15 8 39:
00032 is the diretory of images whose label is buick verano sedan 2012 (see label_map.txt) ,the label buick verano sedan 2012 will be sliced in three parts,the first part is 2012 which is the year of car ,and his index is 15 in car_multi/data/label1.txt, the second part is sedan which is the type of car ,and his index is 8 in car_multi/data/label2.txt,
the third part is buick verano sedan which is the name of car ,and his index is 39 in car/multi/data/label3.txt.)
TRAIN_DATA_ROOT/imagenet_train_lmdb:the lmdb files which produced by images data used in caffe
TRAIN_DATA_ROOT/imagenet_train_label:the label files which produce by images data are used for multi-label training model.
3: the total labels you want to slice,for example car_name,car_type and car_year


step3:make imagenet_mean.binaryproto by imagenet_train_lmdb and imagenet_train_label.

you can use command line:

sh make_imagenet_mean.sh



step4:start training model

you can use command line:

sh train_caffenet.sh


step 5:Use our model to classify a picture

you can use command line to test one image:

sh test_caffenet.sh

explain the parameters:

/root/caffe/build/tools/classification_multilabel:the path of classification_multilabel which produced by caffe
/data1/liangdas/multi_labels/caffe_script/car_multi/data/deploy.prototxt:the deploy.prototxt file is the net model which predict the output
/data1/liangdas/multi_labels/caffe_script/car_multi/data/googLenet_fine_tuning/googLenet_iter_300000.caffemodel:the caffemodel which we have trained
/data1/liangdas/multi_labels/caffe_script/car_multi/data/imagenet_train_mean.binaryproto:the binaryproto file we used to train the model
label3.txt label2.txt label1.txt:the labels files 
/data1/liangdas/multi_labels/caffe_script/car_multi/data/003913.jpg:the images you want to test




step 6:Use our model to classify direcoty of images

you can use out model to examine the accuracy of images data.

before examine the accuracy ,you need to modify classifier.py,the modify context has been given in classifier.py ,
you should replace the classifier.py which is only support single label classification in caffe.

then you can use the file named pycaffe_multilabel.py to test your images data.

the line command :python pycaffe_multilabel.py 

the result log will be produced by this command ,you can see my car_multi/data/result_test_log.txt.



