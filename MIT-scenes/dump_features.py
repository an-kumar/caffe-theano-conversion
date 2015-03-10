'''
This script reads in the BVLC reference network and feature extracts from all the MIT scene data.
'''
import skimage.transform
import os
import argparse
import sys
sys.path.insert(0,'../') # to get the caffe2theano module
import caffe2theano
import skimage.io
import numpy as np


lmodel = caffe2theano.conversion.convert(prototxt='/root/caffe/models/bvlc_reference_caffenet/deploy.prototxt', caffemodel='/root/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
mean_image = caffe2theano.conversion.convert_mean_image('/root/caffe/data/ilsvrc12/imagenet_mean.binaryproto')
mean_image = skimage.transform.resize(mean_image, (3,227,227))
print mean_image.shape
base_dir = '/root/proj/Images/'
out_dir = '/root/proj/MIT_dumped'
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=80)


def process_single_image(X):
	# transpose
        try:
	    X = X.transpose(2,0,1)
        except:
            print X.shape
            return None
	# scale to mean image
	X = skimage.transform.resize(X, mean_image.shape)
	# subtract mean image
	X = X - mean_image
	return X.astype(np.float32)

def process_single_file(filename):
	X = skimage.io.imread(filename)
	return process_single_image(X)

# build label maps

i2l = os.listdir(os.path.join(base_dir,'train'))
if len(i2l) != 67:
	print "PROBLEM, TOO MANY CLASSES!!"

l2i = {i2l[i]:i for i in range(len(i2l))}

# now dump

for split in ['train', 'test']:
	all_arrs = []
	all_ys= []
	full_dir = os.path.join(base_dir,split)
	for label in os.listdir(full_dir):
		y = l2i[label]
                dirpath = os.path.join(full_dir,label)
                files = [os.path.join(dirpath,x) for x in os.listdir(dirpath)]
		arr_list = [process_single_file(f) for f in files]
                arr_list = [x for x in arr_list if x is not None]
                all_arrs += arr_list
                y_list = [y for i in range(len(arr_list))]
                all_ys += y_list
		# arr_list is a list of files. we need to turn them into batches of for the reference caffenet

# 		# y_arr = np.array([y for i in range(len(files))])
# 		# tensor = np.array(arr_list) #4d tensor
# 		# np.save(os.path.join(out_dir,'X_%s_%s' % (label,split)), tensor)
# 		# np.save(os.path.join(out_dir,'y_%s_%s' % (label,split)), y_arr)

# import pickle
# pickle.dump(l2i, open(os.path.join(out_dir, 'label_to_index'),'w'))
	split_all_arrs= [all_arrs[i*10:(i+1)*10] for i in range((len(all_arrs)/10)+1)]
	split_all_ys = [all_ys[i*10:(i+1)*10] for i in range((len(all_ys)/10)+1)]
	all_ys = []
	for spl in split_all_ys[:-1]:
		all_ys += spl

	outs = [lmodel.forward(spl)[0] for spl in split_all_arrs[:-1]]
	total_tensor = np.concatenate(outs, axis=0)
	total_y = np.array(all_ys)
	print total_tensor.shape
	print total_y.shape

	np.save(os.path.join(out_dir,'X_%s' % (split)), total_tensor)
	np.save(os.path.join(out_dir,'y_%s' % (split)), total_y)

