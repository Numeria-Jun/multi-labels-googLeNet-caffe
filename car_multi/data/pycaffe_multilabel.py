#!coding=utf-8
import os
import glob
import numpy as np
import caffe
import datetime
import pdb
#from lib.log_util import create_log
#app_logger = create_log('googlenet_vehcolor_recognize')

class Recogniton(object):
	def __init__(self, googlenet_model_path = './googLenet_fine_tuning/googLenet_iter_300000.caffemodel',
				googlenet_deploy_path = './deploy.prototxt',
				googlenet_mean_path = './mean.npy',
                labels='label3.txt\tlabel2.txt\tlabel1.txt',
				gpu = True,
				gpu_device = 1):
		if(gpu):
			caffe.set_mode_gpu()
			#caffe.set_device(gpu_device)  #must commented, or it would cause error
		else:
			caffe.set_mode_cpu()
		
		self.googlenet_model_path = googlenet_model_path
		self.googlenet_deploy_path = googlenet_deploy_path
	        self.googlenet_mean = np.load(googlenet_mean_path).mean(1).mean(1) #this is a tuple type
		#self.label_id_map_file = label_id_map_file
               #self.googlenet_mean=googlenet_mean_path
	        self.labels=labels
		
	def load_net(self):
		googlenet = caffe.Classifier(self.googlenet_deploy_path, self.googlenet_model_path,mean=self.googlenet_mean,channel_swap=(2,1,0), raw_scale=255, image_dims=(227, 227))

		return googlenet

def googlenet_recognition(googlenet, image):
	try:
		#  Absolute Path
		#input_image = caffe.io.load_image(image)
        
		predictions = googlenet.predict([image])
		return prediction
	except ValueError:
		app_logger.error('caffe.io.load_image ValueError')
		return None
	except IOError:
		app_logger.error('caffe.io.load_image IOError')
		return None
	except Exception, e:
		app_logger.error('googlenet_vehcolor_recognize predict phase failed! %s'%e.message)
		return None

def run_inference_on_image(imageIO, googlenet):
	try:
		#  Absolute Path
		image = caffe.io.load_image(imageIO)
	except ValueError:
		app_logger.error('caffe.io.load_image ValueError')
		return
	except IOError:
		app_logger.error('caffe.io.load_image IOError')
		return 
	try:
		prediction = googlenet_recognition(googlenet, image)
		return prediction
	except ValueError:
		app_logger.error('caffe.io.load_image ValueError')
		return None
	except IOError:
		app_logger.error('caffe.io.load_image IOError')
		return None
	except Exception, e:
		app_logger.error('googlenet_vehcolor_recognize predict phase failed! %s'%e.message)
		return None
	
if __name__=='__main__':
    recognition_obj = Recogniton()  #initial an object
    googlenet = recognition_obj.load_net()
	#ssd_transformer = recognition_obj.ssd_transformer(ssdnet)
	#label_map = resnet.load_label_map()

    start=datetime.datetime.now()
    test_path='/data1/liangdas/multi_labels/caffe_script/car_multi/data/new_label_test.txt'
    test_txt=open(test_path,'r')
    result_txt=open('result_test_log.txt','w')
    test_lines=test_txt.readlines()
    
    image_path=[]
    car_name=[]
    car_year=[]
    car_typ=[]
    
    true_count=0
    total_count=len(test_lines)
    print ('total test images are %d'%total_count)
    for line in test_lines:
        if line:
            line_split=line.split(' ')
            image_full_path='/data1/liangdas/multi_labels/caffe_script/car_multi/data/'+line_split[0]
          # predictions=run_inference_on_image(image_full_path,googlenet)
            image=caffe.io.load_image(image_full_path)
            predictions = googlenet.predict([image])
            proba_name = predictions[0].argmax()
            print ('the predict name number is %d)'%proba_name)
            print ('the year is %d'%int (line_split[-1]))
            proba_type = predictions[1].argmax()
            print ('the predict type number is %d)'%proba_type)
            print ('the type is %d'%int(line_split[2]))
            proba_year = predictions[2].argmax()
            print ('the predict year number is %d)'%proba_year)
            print ('the name is %d'%int (line_split[1]))
            
         #   image_path.append(line_split[0])
            if (int(proba_year)==int(line_split[1]))and(int (proba_type)==int (line_split[2]))and (int (proba_name)==int (line_split[-1])):
                true_count+=1
                print (line +'  true')
                result_txt.write(line +'  true'+'\n')
            else:
                print (line+'  false')
                result_txt.write(line+'  false'+'\n')
    print ('total test images are %d'%total_count)
    result_txt.write('total test images are %d'%total_count+'\n')
    accuracy=float(true_count)/float(total_count)
    result_txt.write('true_count is %d'%true_count+'\n')
    print ('true_count =%d'%true_count)
    print ('accuracy is %.4f'%accuracy)
    result_txt.write('accuracy is %.4f'%accuracy+'\n')
    result_txt.close()
    test_txt.close()
          
    
   # image_path='/data1/liangdas/multi_labels/caffe_script/car_multi/data/003913.jpg'
  #  image=caffe.io.load_image(image_path)
  #  predictions = googlenet.predict([image])
  #  print ('have run classifier.Classify')
  #  
  #  print ('---------- car name---------')
  #  print ('the predict name number is %d)'%proba_name)
  #  proba_name = predictions[0].argmax()
    
  #  print ('---------- car type---------')
  #  print ('the predict type number is %d)'%proba_type)
  #  proba_type = predictions[1].argmax()

  #  print ('----------car year----------')
   # print ('the predict year number is %d)'%proba_year) 
   # proba_year = predictions[2].argmax()
    
   
    
    
    
    end=datetime.datetime.now()
    print (end-start)

