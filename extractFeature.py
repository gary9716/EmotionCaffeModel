import os
import numpy as np
import matplotlib.pyplot as plt
import caffe

imgPath = '../../persona/images/'
N = 200
outputFileName = 'emotionFeature_200'

categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
DEMO_DIR = '.'
cur_net_dir = 'VGG_S_rgb'

caffe.set_mode_gpu()
caffe.set_device(0)

mean_filename=os.path.join(DEMO_DIR,cur_net_dir,'mean.binaryproto')
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

net_pretrained = os.path.join(DEMO_DIR,cur_net_dir,'EmotiW_VGG_S.caffemodel')
net_model_file = os.path.join(DEMO_DIR,cur_net_dir,'deploy.prototxt')

VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

numEmotions = len(categories)
tempBuf = [0 for i in xrange(0,numEmotions,1)]
with open(outputFileName, 'w') as outfile:
	for i in range(N):
		inputImg = caffe.io.load_image(os.path.join(imgPath, '%d.jpg' % (i + 1)))
		prediction = VGG_S_Net.predict([inputImg],oversample=False)
		# print 'predicted category is {0}'.format(categories[prediction.argmax()])
		for index in xrange(0, numEmotions, 1):
			tempBuf[index] = str(prediction[0][index])
		outfile.write('%s\n' % ','.join(tempBuf))